"""Efficient 0.1 m output-head adaptation over a frozen-resolution BEV grid.

The image encoder and 100 x 100 (0.2 m) BEV encoder are unchanged from the
source GVAD + Agri-AMoE + NearFar model.  Only the public semantic-occupancy
decode is lifted to 200 x 200 through a learned 2x residual refiner.
"""

_base_ = ['./farmsim_occ_front3.py']

# The supervision grid is 0.1 m, while the BEV query grid intentionally stays
# at 100 x 100.  The WorldHeadV1 refiner produces the dense 200 x 200 logits.
voxel_size = [0.1, 0.1, 0.2]
occ_size = [200, 200, 25]

model = dict(
    future_pred_head=dict(
        use_agri_amoe_decoder=True,
        agri_amoe_channels=96,
        agri_amoe_use_gradient_energy=True,
        agri_amoe_use_saliency=True,
        agri_amoe_gate_temperature=1.0,
        use_occupancy_2x_refiner=True,
        occupancy_2x_refiner_channels=32,
    ),
    pts_bbox_head=dict(
        bbox_coder=dict(voxel_size=voxel_size),
        transformer=dict(encoder=dict(
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
                        pc_range=[0.0, -10.0, -2.0, 20.0, 10.0, 3.0],
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

work_dir = 'work_dirs/farmsim_occ_front3_0p1_headrefine_gvad_amoe_nearfar'
