"""Runtime compatibility for occupancy training environments.

MMDetection 2.x imports MMCV's generic ``MultiScaleDeformableAttention``
while loading every model registry entry.  FarmSim occupancy configurations
use this repository's ``MSDeformableAttention3D`` instead, so a CUDA-backed
generic MMCV operator is not required merely to import ResNet/FPN.

With ``mmcv-full`` installed, native operators are preserved.  With lightweight
``mmcv`` it supplies a placeholder which fails only if a configuration actually
requests the unsupported generic operator.  Parallel-runtime compatibility
also applies to the external baselines, which use older PyTorch versions.
"""

try:
    import importlib
    import sys
    import types
    try:
        importlib.import_module('mmcv._ext')
    except ModuleNotFoundError:
        class _UnavailableMMCVExtension(types.ModuleType):
            """Lets unused MMCV operators be imported without mmcv-full."""

            __farm_sim_occ_fallback__ = True

            def __getattr__(self, name):
                if name.startswith('__'):
                    raise AttributeError(name)
                if name in ('get_compiler_version', 'get_compiling_cuda_version'):
                    return lambda: 'not built (occupancy-only)'

                def _missing_operator(*args, **kwargs):
                    raise RuntimeError(
                        f'MMCV operator {name} is unavailable in the '
                        'occupancy-only dow2 environment.')

                return _missing_operator

        sys.modules['mmcv._ext'] = _UnavailableMMCVExtension('mmcv._ext')
        # MMCV's optimizer constructor probes for ``mmcv._ext`` using
        # ``pkgutil``.  The lightweight module above is intentionally not a
        # binary extension, so report that no compiled operators exist.
        import mmcv.utils.ext_loader as _ext_loader
        _ext_loader.check_ops_exist = lambda: False

    import mmcv.cnn.bricks.transformer as _transformer

    # MMCV 1.4 calls a private DDP method removed by modern PyTorch.  PyTorch
    # DDP synchronises parameters during construction, so the legacy extra
    # call is safely a no-op for dow2.
    from mmcv.parallel import MMDistributedDataParallel as _MMDDP
    if not hasattr(_MMDDP, '_sync_params'):
        _MMDDP._sync_params = lambda self: None
    # MMCV passes integer GPU IDs, but PyTorch changed this private helper's
    # contract from int (e.g. Torch 2.0) to torch.device (e.g. Torch 2.7).
    # Detect the actual signature rather than imposing dow2's API on COTR
    # and SurroundOcc.  Keep PyTorch's original helper even on module reload.
    import inspect as _inspect
    import torch as _torch
    import mmcv.parallel._functions as _mmcv_parallel_functions
    from torch.nn.parallel._functions import _get_stream as _torch_get_stream
    _torch_stream_uses_device = (
        _inspect.signature(_torch_get_stream).parameters['device'].annotation
        in (_torch.device, 'torch.device'))

    def _dow2_get_stream(device):
        if _torch_stream_uses_device:
            if isinstance(device, int):
                device = (_torch.device('cpu') if device == -1
                          else _torch.device('cuda', device))
            elif isinstance(device, _torch.device) and device.type == 'cuda' and device.index is None:
                device = _torch.device('cuda', _torch.cuda.current_device())
        elif isinstance(device, _torch.device):
            if device.type == 'cpu':
                device = -1
            elif device.type == 'cuda':
                device = (device.index if device.index is not None
                          else _torch.cuda.current_device())
            else:
                raise ValueError(f'Legacy CUDA scatter does not support {device.type}.')
        return _torch_get_stream(device)
    _mmcv_parallel_functions._get_stream = _dow2_get_stream

    if not hasattr(_transformer, 'MultiScaleDeformableAttention'):
        from mmcv.cnn.bricks.registry import ATTENTION
        from mmcv.runner import BaseModule

        @ATTENTION.register_module()
        class MultiScaleDeformableAttention(BaseModule):
            """Import-time compatibility placeholder for unused MMDetection ops."""

            def __init__(self, *args, **kwargs):
                super().__init__(init_cfg=kwargs.pop('init_cfg', None))

            def forward(self, *args, **kwargs):
                raise RuntimeError(
                    'This configuration requests MMCV generic '
                    'MultiScaleDeformableAttention, which is intentionally '
                    'not installed in the occupancy-only dow2 environment.')

        _transformer.MultiScaleDeformableAttention = MultiScaleDeformableAttention
except ImportError:
    # Python also imports sitecustomize outside the training environment.
    pass
