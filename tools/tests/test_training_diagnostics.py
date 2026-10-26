"""Real CPU worker failures and launcher logging; never start model training."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_training_entry_records_uncaught_errors(self):
        tree = ast.parse((ROOT / 'tools/train.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        self.assertTrue(any(isinstance(n, ast.Name) and n.id == 'record' for n in main.decorator_list))
        self.assertEqual(main.body[0].value.func.id, 'configure_worker_diagnostics')

    def _worker_failure(self, mode):
        tree = ast.parse((ROOT / 'tools/train.py').read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'configure_worker_diagnostics')
        with tempfile.TemporaryDirectory(prefix='farmsim_worker_failure_') as tmp:
            root = Path(tmp)
            worker = root / 'worker.py'
            worker.write_text(
                'import faulthandler, os, signal, sys, time\n'
                'from torch.distributed.elastic.multiprocessing.errors import record\n'
                + ast.unparse(helper) + '\n'
                '@record\n'
                'def main():\n'
                '    configure_worker_diagnostics()\n'
                '    if os.environ["RANK"] == "0":\n'
                '        if sys.argv[1] == "signal":\n'
                '            os.kill(os.getpid(), signal.SIGTERM)\n'
                '        raise RuntimeError("injected-worker-failure")\n'
                '    time.sleep(20)\n'
                'main()\n')
            env = dict(os.environ, FARMSIM_TRACE_SIGTERM='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
            result = subprocess.run(
                [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                 '--nproc_per_node=2', '--log-dir', str(root / 'logs'), '--tee', '3',
                 str(worker), mode], env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Root Cause', result.stdout + result.stderr)
            errors = [json.loads(p.read_text()) for p in root.rglob('error.json')]
            stderr = '\n'.join(p.read_text() for p in root.rglob('stderr.log'))
            self.assertIn('[worker-start] pid=', stderr)
            if mode == 'signal':
                self.assertRegex(result.stderr, r'exitcode\s*:\s*-15')
                self.assertIn('worker.py', stderr)  # faulthandler's stack dump
                self.assertEqual(errors, [])  # do not turn a signal into a fake Python error
            else:
                self.assertTrue(any('injected-worker-failure' in json.dumps(e) for e in errors))
                self.assertIn('RuntimeError: injected-worker-failure', stderr)

    def test_sigterm_keeps_signal_exit_and_stack(self):
        self._worker_failure('signal')

    def test_python_exception_gets_a_structured_error_file(self):
        self._worker_failure('exception')

    def test_shell_preserves_stdout_stderr_and_nonzero_exit(self):
        for status in (0, 17):
            with self.subTest(exit_status=status), tempfile.TemporaryDirectory(prefix='farmsim_launch_') as tmp:
                root = Path(tmp)
                fake = root / 'python_capture'
                checkpoint = root / 'image.pth'
                checkpoint.touch()
                fake.write_text(
                    '#!/usr/bin/python3\nimport json, os, sys\n'
                    'if "tools/train.py" in sys.argv:\n'
                    '    print("training-stdout", flush=True)\n'
                    '    print("training-stderr", file=sys.stderr, flush=True)\n'
                    '    with open("captured.json", "w") as f: json.dump(sys.argv[1:], f)\n'
                    '    sys.exit(int(os.environ["FAKE_TRAIN_EXIT"]))\n')
                fake.chmod(0o755)
                env = dict(os.environ, PYTHON_BIN=str(fake), FAKE_TRAIN_EXIT=str(status),
                           CONFIG=str(ROOT / 'projects/configs/farmsim/farmsim_occ_front3_s.py'),
                           PRETRAINED_FROM=str(checkpoint), FARMSIM_TASKS='amoe_cropsample',
                           CAMERA_GEOMETRY='grid_world')
                result = subprocess.run(['bash', str(ROOT / 'tools/train_farmsim_front3_s.sh')],
                                        cwd=root, env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, status, result.stderr)
                command = json.loads((root / 'captured.json').read_text())
                work_dir = root / command[command.index('--work-dir') + 1]
                self.assertIn('--tee', command)
                self.assertIn('--log-dir', command)
                log = (work_dir / 'launcher.log').read_text()
                self.assertIn('training-stdout', log)
                self.assertIn('training-stderr', log)
                self.assertEqual(int((work_dir / 'launcher.exitcode').read_text()), status)


if __name__ == '__main__':
    unittest.main()
