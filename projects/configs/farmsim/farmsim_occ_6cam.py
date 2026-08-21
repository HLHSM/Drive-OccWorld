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
        num_classes=12,
        history_queue_length=2,
        num_pred_height=25,
        use_plan_traj=False,
        bev_h=bev_h_, bev_w=bev_w_, pc_range=point_cloud_range,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        prev_render_neck=dict(sem_norm=False, pred_height=25, num_cls=12),
    ),
    pts_bbox_head=dict(
        bev_h=bev_h_, bev_w=bev_w_, num_classes=12,
        bbox_coder=dict(pc_range=point_cloud_range, num_classes=12),
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        transformer=dict(use_can_bus=False, encoder=dict(pc_range=point_cloud_range,
            transformerlayers=dict(attn_cfgs=[
                dict(type='TemporalSelfAttention', embed_dims=256, num_levels=1),
                dict(type='SpatialCrossAttention', pc_range=point_cloud_range,
                     deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256, num_points=8, num_levels=4),
                     embed_dims=256),
            ]))),
    ),
)

data = dict(
    samples_per_gpu=1, workers_per_gpu=4,
    train=dict(type='FarmSimWorldDataset', ann_file='data/farmsim/splits/train.json',
               queue_length=2, camera_mode='surround', front_only=False, image_size=(640, 360), pipeline=[]),
    val=dict(type='FarmSimWorldDataset', ann_file='data/farmsim/splits/val.json',
             queue_length=2, camera_mode='surround', front_only=False, image_size=(640, 360), pipeline=[], test_mode=True),
    test=dict(type='FarmSimWorldDataset', ann_file='data/farmsim/splits/val.json',
              queue_length=2, camera_mode='surround', front_only=False, image_size=(640, 360), pipeline=[], test_mode=True),
)

load_from = None
find_unused_parameters = True
work_dir = 'work_dirs/farmsim_occ_6cam'
