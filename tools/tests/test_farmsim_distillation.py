"""Numerical/gradient checks for the two FarmSim KD losses (CPU only)."""

import unittest

import sitecustomize  # noqa: F401
import torch
from projects.mmdet3d_plugin.bevformer.detectors.farmsim_distiller import (
    balanced_occupancy_kl, channel_spatial_kl)


class DistillationLossTests(unittest.TestCase):
    def test_class_balance_is_independent_of_background_count(self):
        teacher = torch.tensor([[4., 0.], [0., 4.]]).T.reshape(1, 2, 2, 1, 1)
        student = torch.tensor([[2., 0.], [0., 0.]]).T.reshape(
            1, 2, 2, 1, 1).requires_grad_()
        target = torch.tensor([0, 1]).reshape(1, 2, 1, 1)
        loss, fraction = balanced_occupancy_kl(student, teacher, target)
        indices = torch.tensor([0] * 9 + [1])
        repeated, _ = balanced_occupancy_kl(
            student[:, :, indices], teacher[:, :, indices], target[:, indices])
        torch.testing.assert_close(loss, repeated)
        self.assertEqual(fraction.item(), 1.)
        loss.backward()
        self.assertGreater(student.grad.abs().sum().item(), 0.)

    def test_teacher_mistakes_and_ignore_labels_do_not_distill(self):
        teacher = torch.tensor([3., 0., 3., 0.]).reshape(1, 2, 2, 1, 1)
        student = torch.randn_like(teacher, requires_grad=True)
        target = torch.tensor([1, 255]).reshape(1, 2, 1, 1)
        loss, fraction = balanced_occupancy_kl(student, teacher, target)
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(fraction.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad.abs().sum().item(), 0.)

    def test_teacher_does_not_receive_gradients(self):
        student = torch.randn(2, 6, 3, 2, 2, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        loss, _ = balanced_occupancy_kl(student, teacher, teacher.detach().argmax(1))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_cwd_normalizes_spatially_and_detaches_teacher(self):
        teacher = torch.randn(2, 4, 3, 5, requires_grad=True)
        # Per-channel offsets must not change the spatial probability map.
        student = (teacher.detach() + torch.randn(2, 4, 1, 1)).requires_grad_()
        loss = channel_spatial_kl(student, teacher)
        self.assertLess(abs(loss.item()), 1e-6)
        student = torch.zeros_like(teacher, requires_grad=True)
        loss = channel_spatial_kl(student, teacher)
        self.assertGreater(loss.item(), 0.)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0.)

    def test_cwd_handles_large_backbone_scale_differences(self):
        teacher = torch.randn(2, 4, 3, 5)
        student = (teacher * 1000.0 + 300.0).requires_grad_()
        loss = channel_spatial_kl(student, teacher)
        self.assertLess(abs(loss.item()), 1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        constant = torch.zeros(2, 4, 3, 5, requires_grad=True)
        channel_spatial_kl(constant, teacher).backward()
        self.assertTrue(torch.isfinite(constant.grad).all())


if __name__ == '__main__':
    unittest.main()
