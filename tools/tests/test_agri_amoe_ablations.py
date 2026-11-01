"""CPU checks for independent Agri-AMoE ablations and their launcher configs."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import sitecustomize  # noqa: F401
import torch
from mmcv import Config
from mmdet.models import build_head

from projects.mmdet3d_plugin.bevformer.dense_heads.world_head_v1 import _AgriAMoE3D
from tools.train import apply_farmsim_options, parse_args


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools/train_farmsim_front3_s.sh'


class AgriAMoEAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_channel_and_spatial_switches_are_independent(self):
        feature = torch.randn(2, 8, 3, 4, 5)
        no_channel = _AgriAMoE3D(8, use_channel_attention=False)
        captured = []
        no_channel.energy_gate.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        no_channel(feature)
        expected = no_channel.spatial_gate(torch.cat((
            feature.mean(1, keepdim=True), feature.amax(1, keepdim=True)), 1))
        torch.testing.assert_close(captured[0][:, 1:], expected)
        self.assertIsNone(no_channel.channel_gate)

        no_spatial = _AgriAMoE3D(8, use_spatial_saliency=False)
        captured = []
        no_spatial.energy_gate.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        no_spatial(feature)
        self.assertIsNotNone(no_spatial.channel_gate)
        self.assertIsNone(no_spatial.spatial_gate)
        self.assertEqual(captured[0][:, 1:].count_nonzero().item(), 0)
        enhanced = feature * no_spatial.channel_gate(feature)
        torch.testing.assert_close(captured[0][:, :1],
                                   torch.log1p(no_spatial._gradient_energy(enhanced)))

    def test_energy_switch_preserves_spatial_signal(self):
        module = _AgriAMoE3D(8, use_gradient_energy=False)
        captured = []
        module.energy_gate.register_forward_pre_hook(
            lambda layer, args: captured.append(args[0].detach()))
        module(torch.randn(2, 8, 3, 4, 5))
        self.assertEqual(captured[0][:, :1].count_nonzero().item(), 0)
        self.assertGreater(captured[0][:, 1:].min().item(), 0)

    def test_fixed_fusion_matches_manual_weighted_sum(self):
        feature = torch.randn(2, 8, 3, 4, 5, requires_grad=True)
        for fusion in ('mean', 'global'):
            with self.subTest(fusion=fusion):
                module = _AgriAMoE3D(8, fusion=fusion).eval()
                if fusion == 'global':
                    with torch.no_grad():
                        module.global_logits.copy_(torch.tensor([2., -1., 1.]))
                    weights = module.global_logits.softmax(0)
                else:
                    weights = torch.full((3,), 1. / 3)
                enhanced = feature * module.channel_gate(feature)
                mixed = sum(weight * expert(enhanced) for weight, expert in
                            zip(weights, module.experts.values()))
                expected = torch.relu(enhanced + module.out_norm(mixed))
                actual = module(feature)
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                self.assertIsNone(module.energy_gate)
                self.assertIsNone(module.spatial_gate)
                if fusion == 'global':
                    self.assertGreater(module.global_logits.grad.abs().sum().item(), 0)

    def test_variants_support_training_and_remove_unused_branches(self):
        variants = [dict(), dict(use_channel_attention=False),
                    dict(use_spatial_saliency=False), dict(use_gradient_energy=False),
                    dict(expert_layout='planar'), dict(fusion='mean'),
                    dict(fusion='global'), dict(active_experts=(1, 2)),
                    dict(active_experts=(0, 2)), dict(active_experts=(0, 1))]
        for options in variants:
            with self.subTest(options=options):
                module = _AgriAMoE3D(8, **options).train()
                feature = torch.randn(2, 8, 3, 4, 5, requires_grad=True)
                output = module(feature)
                self.assertEqual(output.shape, feature.shape)
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(feature.grad).all())
                for name, parameter in module.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if 'active_experts' in options:
                    self.assertEqual(set(module.experts),
                                     set(map(str, options['active_experts'])))
                    self.assertEqual(module.energy_gate[-1].out_channels, 2)
                if options.get('expert_layout') == 'planar':
                    self.assertTrue(all(expert.depthwise.kernel_size == (1, 3, 3)
                                        and expert.depthwise.dilation == (1, 1, 1)
                                        for expert in module.experts.values()))

    def test_rejects_invalid_ablation_configs(self):
        for options in (dict(active_experts=()), dict(active_experts=(1, 1)),
                        dict(active_experts=(3,)), dict(fusion='unknown'),
                        dict(expert_layout='unknown')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                _AgriAMoE3D(8, **options)

    def test_launcher_configs_reach_real_head_and_remove_tsa(self):
        with tempfile.TemporaryDirectory(prefix='amoe_ablations_') as tmp:
            root = Path(tmp)
            for name in ('train.json', 'val.json'):
                (root / name).touch()
            env = dict(os.environ, DRY_RUN='1', FARMSIM_TASKS='all',
                       CUDA_VISIBLE_DEVICES='0', NUM_GPUS='1', BATCH_SIZE='2',
                       TOTAL_BATCH_SIZE='24', DATA_ROOT=tmp,
                       TRAIN_ANN_FILE=str(root / 'train.json'),
                       VAL_ANN_FILE=str(root / 'val.json'))
            result = subprocess.run(['bash', str(SCRIPT)], cwd=ROOT, env=env,
                                    text=True, capture_output=True, check=True)
            commands = [shlex.split(line) for line in result.stdout.splitlines()
                        if line.startswith('env ')]
            self.assertEqual(len(commands), 12)
            for command in commands:
                with patch.object(sys, 'argv', command[command.index('tools/train.py'):]):
                    args = parse_args()
                cfg = Config.fromfile(str(ROOT / args.config))
                cfg.merge_from_dict(args.cfg_options)
                with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
                    apply_farmsim_options(cfg, args)
                # A CropSample config must not leak into the uniform controls.
                self.assertEqual(dict(cfg.data.shuffler_sampler),
                                 dict(type='DistributedGroupSampler'))
                encoder = cfg.model.pts_bbox_head.transformer.encoder
                if args.disable_temporal_self_attention:
                    self.assertEqual(encoder.transformerlayers.operation_order,
                                     ('cross_attn', 'norm', 'ffn', 'norm'))
                    self.assertEqual(len(encoder.transformerlayers.attn_cfgs), 1)
                else:
                    self.assertEqual(encoder.transformerlayers.attn_cfgs[0].type,
                                     'TemporalSelfAttention')
                head_cfg = cfg.model.future_pred_head
                head = build_head(head_cfg)
                if head_cfg.use_agri_amoe_decoder:
                    self.assertEqual(head.agri_amoe.active_experts,
                                     tuple(head_cfg.agri_amoe_active_experts))
                    self.assertEqual(head.agri_amoe.fusion, head_cfg.agri_amoe_fusion)
                    self.assertEqual(head.agri_amoe.use_channel_attention,
                                     head_cfg.agri_amoe_use_channel_attention)
                    self.assertEqual(head.agri_amoe.use_spatial_saliency,
                                     head_cfg.agri_amoe_use_spatial_saliency)
                else:
                    self.assertFalse(hasattr(head, 'agri_amoe'))


if __name__ == '__main__':
    unittest.main()
