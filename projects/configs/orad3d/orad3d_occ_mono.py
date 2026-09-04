"""ORAD-3D monocular current-frame semantic occupancy configuration.

The architecture intentionally matches the active FarmSim front3 recipe:
GVAD + ADHR + NearFar BEV, without temporal history or planning.  Only the
camera count, physical grid, vertical resolution and semantic output space are
changed for ORAD-3D.
"""

_base_ = ['../farmsim/farmsim_occ_front3.py']

# Model coordinates: x forward, y right, z up.  ORAD labels originally use
# right/forward/up and are swapped by ORAD3DWorldDataset.
point_cloud_range = [5.7, -25.0, -3.0, 55.7, 25.0, 5.0]
occ_size = [100, 100, 16]
bev_h_ = 100
bev_w_ = 100
orad_num_classes = 9

model = dict(
    turn_on_plan=False,
    predict_trajectory=False,
    future_pred_frame_num=0,
    test_future_frame_num=0,
    point_cloud_range=point_cloud_range,
    bev_h=bev_h_,
    bev_w=bev_w_,
    future_pred_head=dict(
        num_classes=orad_num_classes,
        num_pred_height=16,
        history_queue_length=0,
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        # ADHR's foreground class is ORAD road.  FarmSim crop-only modules
        # remain off; this is the same direct semantic decoder as FarmSim.
        crop_gap_crop_class=1,
        use_crop_gap_refinement=False,
        use_selective_c2f=False,
        use_gap_residual_refiner=False,
        use_dual_hardness_refinement=True,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        prev_render_neck=dict(sem_norm=False, pred_height=16,
                              num_cls=orad_num_classes),
    ),
    pts_bbox_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=orad_num_classes,
        bbox_coder=dict(pc_range=point_cloud_range,
                        num_classes=orad_num_classes),
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        transformer=dict(
            num_cams=1,
            encoder=dict(
                pc_range=point_cloud_range,
                use_nearfar_bev=True,
                nearfar_near_ratio=0.6,
                nearfar_far_stride=2,
                transformerlayers=dict(attn_cfgs=[
                    dict(type='GeometryVisibleAnchorDeformableAttention',
                         embed_dims=256, num_heads=8, num_levels=1,
                         num_points=4, num_bev_queue=2,
                         anchor_grid_height=4, anchor_grid_width=8,
                         use_visibility=True, use_local_deformable=True),
                    dict(type='SpatialCrossAttention', pc_range=point_cloud_range,
                         num_cams=1,
                         deformable_attention=dict(
                             type='MSDeformableAttention3D', embed_dims=256,
                             num_points=8, num_levels=4),
                         embed_dims=256),
                ]),
            ),
        ),
    ),
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(type='ORAD3DWorldDataset',
               ann_file='data/orad3d/splits/train_100.json', data_root=None,
               queue_length=0, image_size=(512, 288), pipeline=[]),
    val=dict(type='ORAD3DWorldDataset', ann_file='data/orad3d/splits/val.json',
             data_root=None, queue_length=0, image_size=(512, 288),
             pipeline=[], test_mode=True),
    test=dict(type='ORAD3DWorldDataset', ann_file='data/orad3d/splits/test.json',
              data_root=None, queue_length=0, image_size=(512, 288),
              pipeline=[], test_mode=True),
)

work_dir = 'work_dirs/orad3d_occ_mono'
