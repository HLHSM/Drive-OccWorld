"""Native 0.1 m TSA + Agri-AMoE over the same 15 x 15 m source ROI.

The 150 x 150 BEV is used by the image-pretrained scratch control and the
frozen-base query-adaptation experiment. Crop-aware sampling belongs to the
0.2 m source pretraining; all target-domain runs share uniform sampling.
"""

_base_ = ['./farmsim_occ_front3_s.py']

point_cloud_range = [0.0, -7.5, -2.0, 15.0, 7.5, 3.0]
voxel_size = [0.1, 0.1, 0.2]
occ_size = [150, 150, 25]
bev_h_ = 150
bev_w_ = 150
plan_grid_conf = dict(
    xbound=[0.0, 15.0, 0.1], ybound=[-7.5, 7.5, 0.1],
    zbound=[-2.0, 3.0, 5.0])

model = dict(
    bev_h=bev_h_, bev_w=bev_w_,
    plan_head=dict(
        bev_h=bev_h_, bev_w=bev_w_, plan_grid_conf=plan_grid_conf,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_)),
    future_pred_head=dict(
        bev_h=bev_h_, bev_w=bev_w_, history_queue_length=0,
        use_agri_amoe_decoder=True, agri_amoe_channels=96,
        agri_amoe_use_gradient_energy=True, agri_amoe_use_saliency=True,
        agri_amoe_gate_temperature=1.0,
        use_dual_hardness_refinement=False,
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_)),
    pts_bbox_head=dict(
        bev_h=bev_h_, bev_w=bev_w_,
        bbox_coder=dict(voxel_size=voxel_size),
        positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_),
        transformer=dict(encoder=dict(use_nearfar_bev=False))),
)

data = dict(
    train=dict(queue_length=0, future_pred_frame_num=0, future_traj_frame_num=0,
               predict_trajectory=False, bev_size=(bev_h_, bev_w_),
               image_size=(512, 288)),
    val=dict(queue_length=0, future_pred_frame_num=0, future_traj_frame_num=0,
             predict_trajectory=False, bev_size=(bev_h_, bev_w_),
             image_size=(512, 288)),
    test=dict(queue_length=0, future_pred_frame_num=0, future_traj_frame_num=0,
              predict_trajectory=False, bev_size=(bev_h_, bev_w_),
              image_size=(512, 288)),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
)

work_dir = 'work_dirs/farmsim_occ_front3_s_0p1_tsa_amoe'
