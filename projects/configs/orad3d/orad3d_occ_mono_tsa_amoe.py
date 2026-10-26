"""Official ORAD grid/classes, with the new TSA + Agri-AMoE source model.

Only the source FarmSim ROI was reduced. Keep ORAD's 100 x 100 x 16 grid
and nine official classes so test and test_farm retain their original meaning.
"""

_base_ = ['./orad3d_occ_mono.py']

model = dict(
    future_pred_head=dict(
        use_agri_amoe_decoder=True, agri_amoe_channels=96,
        agri_amoe_use_gradient_energy=True, agri_amoe_use_saliency=True,
        agri_amoe_gate_temperature=1.0,
        use_dual_hardness_refinement=False),
    pts_bbox_head=dict(transformer=dict(encoder=dict(
        use_nearfar_bev=False,
        transformerlayers=dict(attn_cfgs=[
            dict(type='TemporalSelfAttention', embed_dims=256, num_levels=1),
            dict(type='SpatialCrossAttention',
                 pc_range=[5.7, -25.0, -3.0, 55.7, 25.0, 5.0], num_cams=1,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D', embed_dims=256,
                     num_points=8, num_levels=4), embed_dims=256),
        ])))),
)

data = dict(
    train=dict(queue_length=0, future_pred_frame_num=0, future_traj_frame_num=0,
               predict_trajectory=False, bev_size=(100, 100)),
    val=dict(queue_length=0, future_pred_frame_num=0, future_traj_frame_num=0,
             predict_trajectory=False, bev_size=(100, 100)),
    test=dict(queue_length=0, future_pred_frame_num=0, future_traj_frame_num=0,
              predict_trajectory=False, bev_size=(100, 100)),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
)

work_dir = 'work_dirs/orad3d_occ_mono_tsa_amoe'
