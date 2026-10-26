"""75 x 75 frozen TSA/Agri-AMoE encoder, learned 150 x 150 output refiner."""

_base_ = ['./farmsim_occ_front3_s_0p1_tsa_amoe.py']

# The output label grid stays 150 x 150 x 25. Only the BEV query grid and
# source decoder revert to their 0.2 m shape; the 2x head supplies dense logits.
bev_h_ = 75
bev_w_ = 75
plan_grid_conf = dict(
    xbound=[0.0, 15.0, 0.2], ybound=[-7.5, 7.5, 0.2],
    zbound=[-2.0, 3.0, 5.0])

model = dict(
    bev_h=bev_h_, bev_w=bev_w_,
    plan_head=dict(
        bev_h=bev_h_, bev_w=bev_w_, plan_grid_conf=plan_grid_conf,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_)),
    future_pred_head=dict(
        bev_h=bev_h_, bev_w=bev_w_,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        use_occupancy_2x_refiner=True, occupancy_2x_refiner_channels=32),
    pts_bbox_head=dict(
        bev_h=bev_h_, bev_w=bev_w_,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_)),
)

data = dict(
    train=dict(bev_size=(bev_h_, bev_w_)),
    val=dict(bev_size=(bev_h_, bev_w_)),
    test=dict(bev_size=(bev_h_, bev_w_)),
)

work_dir = 'work_dirs/farmsim_occ_front3_s_0p1_headrefine_tsa_amoe'
