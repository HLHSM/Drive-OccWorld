_base_ = ['./farmsim_occ_front3.py']

# Exact 15 m x 15 m forward ROI at the source 0.2 m voxel pitch.
# The y boundaries fall halfway through source voxels, so the dataset samples
# labels at target voxel centers via nearest neighbor for all three splits.
point_cloud_range = [0.0, -7.5, -2.0, 15.0, 7.5, 3.0]
occ_size = [75, 75, 25]
bev_h_ = 75
bev_w_ = 75
plan_grid_conf = dict(
    xbound=[0.0, 15.0, 0.2],
    ybound=[-7.5, 7.5, 0.2],
    zbound=[-2.0, 3.0, 5.0],
)

model = dict(
    point_cloud_range=point_cloud_range,
    bev_h=bev_h_, bev_w=bev_w_,
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
        transformer=dict(encoder=dict(
            pc_range=point_cloud_range,
            transformerlayers=dict(attn_cfgs=[
                dict(type='TemporalSelfAttention', embed_dims=256, num_levels=1),
                dict(type='SpatialCrossAttention', pc_range=point_cloud_range,
                     deformable_attention=dict(type='MSDeformableAttention3D',
                                               embed_dims=256, num_points=8,
                                               num_levels=4),
                     embed_dims=256),
            ]))),
    ),
)

data = dict(
    train=dict(occupancy_target_range=point_cloud_range),
    val=dict(occupancy_target_range=point_cloud_range),
    test=dict(occupancy_target_range=point_cloud_range),
)

work_dir = 'work_dirs/farmsim_occ_front3_s'
