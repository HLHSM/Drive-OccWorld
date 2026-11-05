"""Panoptic supervision, matching, fusion and metric contracts."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sitecustomize  # noqa
import numpy as np
import torch
from mmcv import Config
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.bevformer.dense_heads.panoptic_world_head import PanopticMaskDecoder, linear_sum_assignment
from projects.mmdet3d_plugin.datasets.farmsim_panoptic_dataset import FarmSimPanopticDataset, INSTANCE_IGNORE
from projects.mmdet3d_plugin.core.evaluation.panoptic_metrics import panoptic_counts, panoptic_summary


class PanopticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_flashocc_manifest_and_per_sequence_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ['crop_b/seq2','crop_a/seq1']
            for path,n in zip(paths,[5,2]):
                meta = root/path/'meta'
                meta.mkdir(parents=True)
                for i in reversed(range(n)):
                    (meta/f'{i:06d}.json').write_text('{}')
            manifest = root/'train.json'
            manifest.write_text(json.dumps(dict(data_root=str(root),sequences=paths)))
            ds = FarmSimPanopticDataset(str(manifest),queue_length=0,sample_stride=2,
                occupancy_target_range=[-20,-10,-2,20,10,2])
            pairs = [(ds.sequences[s]['path'],ds.sequences[s]['frame_ids'][i]) for s,i in ds.samples]
            self.assertEqual(pairs,[(paths[0],'000000'),(paths[0],'000002'),
                                    (paths[0],'000004'),(paths[1],'000000')])
            self.assertEqual([s['path'] for s in ds.sequences],paths)
            cfg = Config.fromfile('projects/configs/farmsim/farmsim_panoptic_6cam.py')
            self.assertEqual(cfg.occ_size,[200,100,25])
            self.assertEqual(cfg.point_cloud_range,[-20,-10,-2,20,10,3])
            self.assertEqual(cfg.model.future_pred_head.num_pred_height,25)
            self.assertEqual(cfg.model.future_pred_head.prev_render_neck.pred_height,25)
            self.assertEqual((cfg.model.bev_h,cfg.model.bev_w),(100,200))
            self.assertEqual(cfg.data.train.sample_stride,2)
            self.assertEqual(cfg.data.val.sample_stride,1)
            self.assertEqual(cfg.data.test.sample_stride,1)
            self.assertFalse(cfg.data.train.allow_unknown_padding)
            self.assertEqual(cfg.data.train.ann_file,'/data/HL/FlashOCC/data/simdata/splits/train.json')

    def test_chunked_attention_preserves_values_and_gradients(self):
        from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import chunked_deformable_attention
        from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
        torch.manual_seed(2)
        shapes = torch.tensor([[3,4],[2,2]])
        value = torch.randn(2,16,2,4,requires_grad=True)
        locations = torch.rand(2,11,2,2,3,2,requires_grad=True)
        weights = torch.randn(2,11,2,2,3,requires_grad=True)
        full = multi_scale_deformable_attn_pytorch(value,shapes,locations,weights)
        chunked = chunked_deformable_attention(value,shapes,locations,weights,3)
        torch.testing.assert_close(full,chunked)
        grad_full = torch.autograd.grad(full.square().sum(),(value,locations,weights),retain_graph=True)
        grad_chunked = torch.autograd.grad(chunked.square().sum(),(value,locations,weights))
        for a,b in zip(grad_full,grad_chunked):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)

    def test_hungarian_matches_exhaustive_rectangular_optimum(self):
        rng = np.random.RandomState(2)
        for nq, ng in [(4, 3), (3, 3), (5, 1)]:
            for _ in range(8):
                cost = rng.randn(nq, ng)
                q, g = linear_sum_assignment(cost)
                expected = min(sum(cost[p[j], j] for j in range(ng))
                               for p in itertools.permutations(range(nq), ng))
                self.assertAlmostEqual(cost[q, g].sum(), expected)

    def test_native_taxonomy_axes_and_outside_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root/'s'
            (seq/'occupancy').mkdir(parents=True)
            (seq/'occupancy_schema.json').write_text(json.dumps(dict(panoptic=True,
                semantic_taxonomy='farmsim_tree_v2', region_mode='surround', regions=[dict(
                    id='surround', min_m_xyz=[-20,-10,-2], max_m_xyz=[20,10,3], dimensions_xyz=[200,100,25])])) )
            sem = np.zeros((25,100,200), dtype=np.uint8)
            inst = np.zeros_like(sem, dtype=np.uint32)
            sem[2,3,4], inst[2,3,4] = 10, 77
            sem[7,8,9], inst[7,8,9] = 8, 88
            valid = np.ones_like(sem)
            valid[7,8,9] = 0
            np.savez(seq/'occupancy/000001.npz', occupancy=sem, occupancy_instances=inst, occupancy_valid=valid)
            manifest = root/'train.json'
            manifest.write_text(json.dumps(dict(sequences=[dict(path='s', frame_ids=['000001'])])))
            ds = FarmSimPanopticDataset(str(manifest), data_root=str(root), queue_length=0,
                camera_mode='surround', occupancy_target_range=[-20,-20,-2,20,20,2], allow_unknown_padding=True)
            s, i = ds._load_panoptic(ds.sequences[0], '000001')
            self.assertEqual(tuple(s.shape), (200,200,20))
            self.assertEqual(s[4,53,2].item(), 10)
            self.assertEqual(i[4,53,2].item(), 77)
            self.assertTrue((s[:,:50] == 255).all())
            self.assertTrue((s[:,150:] == 255).all())
            self.assertEqual(i[9,58,7].item(), INSTANCE_IGNORE)
            ds.allow_unknown_padding = False
            ds._panoptic_grid_cache.clear()
            with self.assertRaisesRegex(ValueError, 'exceeds source'):
                ds._load_panoptic(ds.sequences[0], '000001')

    def test_mask_losses_batch_empty_tiny_ignore_and_id_permutation(self):
        torch.manual_seed(3)
        decoder = PanopticMaskDecoder(8, hidden=16, num_queries=4, num_layers=1,
                    memory_shape=(2,2,2), sample_points=20, points_per_instance=3)
        volume = torch.randn(2,8,3,4,5, requires_grad=True)
        sem = torch.zeros(2,3,4,5, dtype=torch.long)
        ins = torch.zeros_like(sem)
        sem[0,0,0,0], ins[0,0,0,0] = 8, 999
        sem[0,1,2,2:4], ins[0,1,2,2:4] = 9, 7
        sem[:,:,0,4] = 255
        outputs = decoder(volume)
        torch.manual_seed(9)
        losses = decoder.loss(outputs, sem, ins)
        renumbered = ins.clone()
        renumbered[ins == 999], renumbered[ins == 7] = 12, 9000
        torch.manual_seed(9)
        other = decoder.loss(outputs, sem, renumbered)
        for k in losses:
            torch.testing.assert_close(losses[k], other[k])
            self.assertTrue(torch.isfinite(losses[k]))
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(volume.grad).all())
        self.assertGreater(volume.grad.abs().sum().item(), 0)
        for p in decoder.parameters():
            self.assertIsNotNone(p.grad)
        blank = decoder.loss(decoder(volume.detach()), torch.full_like(sem,255), ins)
        self.assertTrue(all(v.item() == 0 for v in blank.values()))

    def test_query_capacity_fails_instead_of_dropping_gt(self):
        decoder = PanopticMaskDecoder(8, hidden=16, num_queries=1, num_layers=1,
                                     memory_shape=(1,1,2), sample_points=2)
        sem = torch.full((1,1,1,2),9)
        ins = torch.tensor([[[[1,2]]]])
        with self.assertRaisesRegex(ValueError,'exceed'):
            decoder.loss(decoder(torch.randn(1,8,1,1,2)),sem,ins)

    def test_fusion_class_consistency_and_missing_query(self):
        decoder = PanopticMaskDecoder(8, hidden=16, num_queries=2, num_layers=1)
        outputs = dict(features=torch.ones(1,16,1,1,3), embeddings=torch.ones(1,2,16),
                       classes=torch.full((1,2,7),-20.))
        outputs['classes'][0,0,4] = 20  # other_obstacle
        outputs['classes'][0,1,6] = 20  # no-object
        sem = torch.tensor([[[[0,9,8]]]])
        pred = decoder.predict(outputs,sem,chunk_size=1)
        self.assertEqual(pred.flatten().tolist(),[0,1,0])

    def test_pq_renumbering_merge_false_positive_void_and_rank_sum(self):
        gt = np.array([9,9,0,9,9,255])
        gi = np.array([1,1,0,2,2,INSTANCE_IGNORE])
        pi = np.array([77,77,0,88,88,8])
        pred = np.array([9,9,0,9,9,9])
        counts = panoptic_counts(pred,pi,gt,gi)
        self.assertEqual(panoptic_summary(counts)['panoptic_PQ_th'],1.)
        merged = panoptic_counts(pred,np.ones(6),gt,gi)
        self.assertEqual(merged[9,1],0) # exactly .5 is not a match
        self.assertEqual(merged[9,2],1)
        pred[2] = 9
        pi[2] = 900
        fp = panoptic_counts(pred,pi,gt,gi)
        self.assertEqual(fp[9,2],1)
        unassigned = panoptic_counts(np.array([9,9]),np.array([0,0]),np.array([9,9]),np.array([8,8]))
        np.testing.assert_array_equal(unassigned[9],[0,0,1,1])
        doubled = panoptic_summary([counts,counts])
        self.assertEqual(doubled['panoptic_PQ_th'],1.)
        self.assertEqual(doubled['panoptic_support_9'],4.)

    def test_model_public_axes_and_batch_targets(self):
        cfg = Config.fromfile('projects/configs/farmsim/farmsim_panoptic_6cam.py')
        cfg.model.bev_h, cfg.model.bev_w = 3,4
        cfg.model.future_pred_head.update(bev_h=3,bev_w=4,num_pred_height=3,agri_amoe_channels=8)
        cfg.model.future_pred_head.instance_decoder.update(hidden=16,num_queries=4,num_layers=1,memory_shape=(2,2,2))
        model = build_model(cfg.model)
        model.obtain_ref_bev = lambda img,meta,prev: torch.randn(img.shape[0],12,256,requires_grad=True)
        images = torch.randn(2,1,6,3,16,16)
        metas = [[{}],[{}]]
        sem = torch.zeros(2,1,4,3,3,dtype=torch.long)
        ins = torch.zeros_like(sem)
        sem[0,0,1,2,0],ins[0,0,1,2,0] = 9,5
        losses = model.forward_train(img=images,img_metas=metas,segmentation=sem,instance=ins)
        sum(losses.values()).backward()
        self.assertGreater(model.future_pred_head.agri_amoe_lift.weight.grad.abs().sum().item(),0)
        model._return_prediction_artifacts = True
        model.eval()
        expected = sem[:,0].permute(0,3,2,1)
        logits = torch.full((2,11,3,3,4),-20.)
        logits.scatter_(1,expected.unsqueeze(1),20.)
        model.future_pred_head.forward_panoptic = lambda bev: dict(semantic=logits,instance={})
        model.future_pred_head.instance_decoder.predict = lambda outputs, labels: ins[:,0].permute(0,3,2,1)
        output = model(return_loss=False,img=images,img_metas=metas,segmentation=sem,instance=ins,sample_idx=[7,8])
        self.assertEqual(output['hist_for_iou'][9,9],1)
        self.assertEqual(panoptic_summary(output['panoptic_counts'])['panoptic_PQ_th'],1.)
        self.assertEqual(output['hist_for_iou'].sum(),72)
        self.assertEqual(len(output['prediction_artifacts']),2)
        artifact = output['prediction_artifacts'][0]
        self.assertEqual(artifact['current_instance_gt'][1,2,0],5)
        self.assertEqual(artifact['current_pred'].shape,(4,3,3))
        padded = model(return_loss=False,img=images,img_metas=metas,segmentation=sem,instance=ins,panoptic_eval_include=[True,False])
        self.assertEqual(padded['hist_for_iou'].sum(),36)
        self.assertEqual(padded['panoptic_counts'][9,1],1)


if __name__ == '__main__':
    unittest.main()
