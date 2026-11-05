_base_ = ['./farmsim_occ_6cam.py']

# 40 x 20 x 5 m prediction volume, fully inside the source labels.
# BEVFormer uses H=Y and W=X; public targets use [X,Y,Z].
point_cloud_range = [-20., -10., -2., 20., 10., 3.]
occ_size = [200, 100, 25]
bev_h_ = 100
bev_w_ = 200
data_root = '/data/HL/SimData-panoptic/SimData-panoptic'
split_dir = '/data/HL/FlashOCC/data/simdata/splits'
model = dict(
    type='FarmSimPanoptic', point_cloud_range=point_cloud_range,
    bev_h=bev_h_, bev_w=bev_w_, future_pred_frame_num=0, test_future_frame_num=0,
    turn_on_plan=False, predict_trajectory=False,
    future_pred_head=dict(
        type='PanopticWorldHead', num_classes=11, num_pred_height=25,
        bev_h=bev_h_, bev_w=bev_w_, pc_range=point_cloud_range,
        history_queue_length=0, use_agri_amoe_decoder=True, agri_amoe_channels=96,
        use_plan_traj=False,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        prev_render_neck=dict(sem_norm=False, pred_height=25, num_cls=11),
        instance_decoder=dict(hidden=64, num_queries=256, num_layers=3,
                              memory_shape=(5, 25, 25), sample_points=4096,
                              points_per_instance=16, score_threshold=.3)),
    pts_bbox_head=dict(
        bev_h=bev_h_, bev_w=bev_w_, num_classes=11,
        bbox_coder=dict(pc_range=point_cloud_range, num_classes=11),
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        transformer=dict(encoder=dict(
            pc_range=point_cloud_range,
            transformerlayers=dict(attn_cfgs=[
                dict(type='TemporalSelfAttention', embed_dims=256, num_levels=1),
                dict(type='SpatialCrossAttention', pc_range=point_cloud_range,
                     deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256,
                                               num_points=8, num_levels=4, query_chunk_size=1024), embed_dims=256)])))),
)
data = dict(
    samples_per_gpu=1, workers_per_gpu=4,
    train=dict(type='FarmSimPanopticDataset', data_root=data_root, ann_file=split_dir+'/train.json',
               sample_stride=2,
               queue_length=0, camera_mode='surround', front_only=False, camera_geometry='grid_world',
               image_size=(512, 288), occupancy_target_range=point_cloud_range, allow_unknown_padding=False),
    val=dict(type='FarmSimPanopticDataset', data_root=data_root, ann_file=split_dir+'/val.json', sample_stride=1,
             queue_length=0, camera_mode='surround', front_only=False, camera_geometry='grid_world',
             image_size=(512, 288), occupancy_target_range=point_cloud_range, allow_unknown_padding=False),
    test=dict(type='FarmSimPanopticDataset', data_root=data_root, ann_file=split_dir+'/test.json', sample_stride=1,
              queue_length=0, camera_mode='surround', front_only=False, camera_geometry='grid_world',
              image_size=(512, 288), occupancy_target_range=point_cloud_range, allow_unknown_padding=False),
)
work_dir = 'work_dirs/panoptic_6cam'
