_base_ = ['../e2e/MMO_MSO_with_plan_fully_decoupled.py']

# UE5 FarmSim: x forward, y right, z up; full surrounding 40 x 20 x 5 m.
point_cloud_range = [-20.0, -10.0, -2.0, 20.0, 10.0, 3.0]
occ_size = [200, 100, 25]
bev_h_ = 200
bev_w_ = 100

model = dict(
    turn_on_plan=False,
    predict_trajectory=False,
    future_pred_frame_num=0,
    test_future_frame_num=0,
    point_cloud_range=point_cloud_range,
    bev_h=bev_h_, bev_w=bev_w_,
    future_pred_head=dict(
        num_classes=6,
        history_queue_length=2,
        num_pred_height=25,
        use_plan_traj=False,
        use_row_topology=False,
        row_topology_loss_weight=0.1,
        # Crop/free gap supervision and selected 2x2 BEV subqueries are
        # disabled by default, retaining the original dense baseline.
        use_crop_gap_refinement=False,
        crop_gap_boundary_loss_weight=0.5,
        crop_gap_free_loss_weight=0.25,
        crop_gap_alpha=3.0,
        crop_gap_sigma=1.5,
        crop_gap_radius=4,
        use_selective_c2f=False,
        c2f_active_ratio=0.25,
        c2f_channels=128,
        # Training-only agricultural dual-hardness refinement. It preserves
        # the baseline inference graph while mining uncertain crop/free voxels.
        use_dual_hardness_refinement=False,
        dual_hardness_active_ratio=0.04,
        dual_hardness_gap_ratio=0.5,
        dual_hardness_channels=128,
        dual_hardness_local_scale=0.25,
        dual_hardness_gap_boost=0.5,
        dual_hardness_loss_weight=0.5,
        dual_hardness_distill_weight=0.1,
        dual_hardness_ema_decay=0.99,
        # COTR-inspired fixed semantic hierarchy: free, crop, other occupied.
        use_fixed_group_decoder=False,
        group_decoder_loss_weight=0.3,
        group_decoder_prior_scale=1.0,
        # End-to-end agricultural coarse-to-refinement proposals.  Both are
        # disabled by default and use the direct BEV semantic decoder.
        use_gap_residual_refiner=False,
        gap_refiner_channels=24,
        gap_refiner_blocks=3,
        gap_refiner_coarse_loss_weight=0.15,
        gap_refiner_boundary_loss_weight=0.25,
        gap_refiner_gap_loss_weight=0.5,
        gap_refiner_crop_loss_weight=0.5,
        gap_refiner_use_bev_feature=True,
        gap_refiner_use_image_features=False,
        gap_refiner_image_active_ratio=0.08,
        gap_refiner_image_channels=24,
        gap_refiner_image_levels=2,
        gap_refiner_image_crop_ratio=0.5,
        bev_h=bev_h_, bev_w=bev_w_, pc_range=point_cloud_range,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        prev_render_neck=dict(sem_norm=False, pred_height=25, num_cls=6),
    ),
    pts_bbox_head=dict(
        bev_h=bev_h_, bev_w=bev_w_, num_classes=6,
        bbox_coder=dict(pc_range=point_cloud_range, num_classes=6),
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        transformer=dict(use_can_bus=False, encoder=dict(
            pc_range=point_cloud_range,
            use_nearfar_bev=False,
            nearfar_near_ratio=0.6,
            nearfar_far_stride=2,
            transformerlayers=dict(attn_cfgs=[
                dict(type='TemporalSelfAttention', embed_dims=256, num_levels=1),
                dict(type='SpatialCrossAttention', pc_range=point_cloud_range,
                     deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256, num_points=8, num_levels=4),
                     embed_dims=256),
            ]))),
    ),
    # The base nuScenes configuration enables DCNv2 in the last ResNet
    # stages.  FarmSim occupancy-only dow2 uses the stock PyTorch ResNet
    # operators, so no CUDA extension/toolkit build is required.
    img_backbone=dict(dcn=None, stage_with_dcn=(False, False, False, False)),
)

data = dict(
    samples_per_gpu=1, workers_per_gpu=4,
    train=dict(type='FarmSimWorldDataset', ann_file='data/farmsim/splits/train.json', data_root=None,
               queue_length=2, camera_mode='surround', front_only=False, image_size=(640, 360), pipeline=[]),
    val=dict(type='FarmSimWorldDataset', ann_file='data/farmsim/splits/val.json', data_root=None,
             queue_length=2, camera_mode='surround', front_only=False, image_size=(640, 360), pipeline=[], test_mode=True),
    test=dict(type='FarmSimWorldDataset', ann_file='data/farmsim/splits/val.json', data_root=None,
              queue_length=2, camera_mode='surround', front_only=False, image_size=(640, 360), pipeline=[], test_mode=True),
)

load_from = None
find_unused_parameters = True
work_dir = 'work_dirs/farmsim_occ_6cam'
# TensorBoard is optional.  Keep the minimal dow2 environment free of it and
# retain the standard text log for loss/learning-rate monitoring.
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
