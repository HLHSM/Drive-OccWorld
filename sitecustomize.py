"""Runtime compatibility for the occupancy-only ``dow2`` environment.

MMDetection 2.x imports MMCV's generic ``MultiScaleDeformableAttention``
while loading every model registry entry.  FarmSim occupancy configurations
use this repository's ``MSDeformableAttention3D`` instead, so a CUDA-backed
generic MMCV operator is not required merely to import ResNet/FPN.

When ``mmcv-full`` is present this module is a no-op.  With the lightweight
``mmcv`` package it supplies an explicit placeholder which fails only if a
configuration actually requests the unsupported generic operator.
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
    # MMCV 1.4 passes integer GPU IDs to PyTorch's scatter helper; PyTorch
    # 2.7 expects ``torch.device``.  Preserve the old caller contract.
    import torch as _torch
    import mmcv.parallel._functions as _mmcv_parallel_functions
    _torch_get_stream = _mmcv_parallel_functions._get_stream
    def _dow2_get_stream(device):
        if isinstance(device, int):
            device = _torch.device('cuda', device)
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
