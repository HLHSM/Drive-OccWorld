# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch)

    def before_train_iter(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if hasattr(model, 'set_distillation_progress'):
            model.set_distillation_progress(
                runner.epoch + runner.inner_iter / max(len(runner.data_loader), 1))
