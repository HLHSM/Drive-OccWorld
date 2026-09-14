"""0.1 m front3 AgriOcc configuration for resolution-transfer experiments.

The supervision region remains the front 20 m by 20 m by 5 m cuboid, but
the horizontal voxel size changes from 0.2 m to 0.1 m.  Consequently the
front BEV grid is 200 x 200 and the height axis remains 25 bins at 0.2 m.
The three research modules are fixed here so a converted 0.2 m checkpoint and
the no-occupancy-pretraining control instantiate exactly the same model.
"""

_base_ = ['./farmsim_occ_front3.py']

point_cloud_range = [0.0, -10.0, -2.0, 20.0, 10.0, 3.0]
voxel_size = [0.1, 0.1, 0.2]
occ_size = [200, 200, 25]
bev_h_ = 200
bev_w_ = 200
plan_grid_conf = dict(
    xbound=[0.0, 20.0, 0.1],
    ybound=[-10.0, 10.0, 0.1],
    zbound=[-2.0, 3.0, 5.0],
)

model = dict(
    point_cloud_range=point_cloud_range,
    bev_h=bev_h_,
    bev_w=bev_w_,
    plan_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        plan_grid_conf=plan_grid_conf,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
    ),
    future_pred_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        # Agricultural structure-adaptive mixture-of-experts occupancy decode.
        use_agri_amoe_decoder=True,
        agri_amoe_channels=96,
        agri_amoe_use_gradient_energy=True,
        agri_amoe_use_saliency=True,
        agri_amoe_gate_temperature=1.0,
    ),
    pts_bbox_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        bbox_coder=dict(pc_range=point_cloud_range, voxel_size=voxel_size),
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        transformer=dict(encoder=dict(
            pc_range=point_cloud_range,
            # Keep the same dense-near / stride-2 far-query policy as the
            # source 0.2 m GVAD + Agri-AMoE + NearFar checkpoint.
            use_nearfar_bev=True,
            nearfar_near_ratio=0.6,
            nearfar_far_stride=2,
            transformerlayers=dict(
                type='BEVFormerLayerV2',
                attn_cfgs=[
                    dict(
                        type='GeometryVisibleAnchorDeformableAttention',
                        embed_dims=256,
                        num_levels=1,
                        num_heads=8,
                        anchor_grid_height=4,
                        anchor_grid_width=8,
                        use_visibility=True,
                        use_local_deformable=True,
                    ),
                    dict(
                        type='SpatialCrossAttention',
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=256,
                            num_points=8,
                            num_levels=4,
                        ),
                        embed_dims=256,
                    ),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'),
            ),
        )),
    ),
)

data = dict(
    train=dict(bev_size=(bev_h_, bev_w_)),
    val=dict(bev_size=(bev_h_, bev_w_)),
    test=dict(bev_size=(bev_h_, bev_w_)),
)

work_dir = 'work_dirs/farmsim_occ_front3_0p1_gvad_amoe_nearfar'
