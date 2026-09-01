_base_ = ['./farmsim_occ_6cam.py']

# Three front RGB cameras and only x in [0,20] m are supervised/predicted.
point_cloud_range = [0.0, -10.0, -2.0, 20.0, 10.0, 3.0]
occ_size = [100, 100, 25]
bev_h_ = 100
bev_w_ = 100
plan_grid_conf = dict(
    xbound=[0.0, 20.0, 0.2],
    ybound=[-10.0, 10.0, 0.2],
    zbound=[-2.0, 3.0, 5.0],
)

model = dict(
    point_cloud_range=point_cloud_range,
    bev_h=bev_h_, bev_w=bev_w_,
    # The inherited IR-WM E2E planner is also applied to the front3 BEV.
    # Its input has 100 x 100 queries here, rather than the source config's
    # 200 x 200 surround-view BEV.
    plan_head=dict(
        bev_h=bev_h_, bev_w=bev_w_,
        plan_grid_conf=plan_grid_conf,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
    ),
    future_pred_head=dict(
        bev_h=bev_h_, bev_w=bev_w_, pc_range=point_cloud_range,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
    ),
    pts_bbox_head=dict(
        bev_h=bev_h_, bev_w=bev_w_,
        bbox_coder=dict(pc_range=point_cloud_range),
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        transformer=dict(encoder=dict(pc_range=point_cloud_range,
            transformerlayers=dict(attn_cfgs=[
                dict(type='TemporalSelfAttention', embed_dims=256, num_levels=1),
                dict(type='SpatialCrossAttention', pc_range=point_cloud_range,
                     deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256, num_points=8, num_levels=4),
                     embed_dims=256),
            ]))),
    ),
)

data = dict(
    train=dict(type='FarmSimWorldDataset', ann_file='data/farmsim/splits/train.json', data_root=None,
               queue_length=2, camera_mode='front', front_only=True, image_size=(640, 360), pipeline=[]),
    val=dict(type='FarmSimWorldDataset', ann_file='data/farmsim/splits/val.json', data_root=None,
             queue_length=2, camera_mode='front', front_only=True, image_size=(640, 360), pipeline=[], test_mode=True),
    test=dict(type='FarmSimWorldDataset', ann_file='data/farmsim/splits/val.json', data_root=None,
              queue_length=2, camera_mode='front', front_only=True, image_size=(640, 360), pipeline=[], test_mode=True),
)

work_dir = 'work_dirs/farmsim_occ_front3'
