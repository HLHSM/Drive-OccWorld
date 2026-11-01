_base_ = ['./farmsim_occ_front3_s.py']

# Same three front views, 0.2 m voxels and forward 15 x 15 x 5 m ROI.
# Panoptic exports supply semantic occupancy supervision to the existing head.
data_root = '/data/HL/SimData-panoptic/SimData-panoptic'
split_dir = 'data/simdata_panoptic/splits'
data = dict(
    train=dict(data_root=data_root, ann_file=split_dir + '/train.json',
               queue_length=0, camera_geometry='grid_world', image_size=(512, 288)),
    val=dict(data_root=data_root, ann_file=split_dir + '/val.json',
             queue_length=0, camera_geometry='grid_world', image_size=(512, 288)),
    test=dict(data_root=data_root, ann_file=split_dir + '/test.json',
              queue_length=0, camera_geometry='grid_world', image_size=(512, 288)),
)
model = dict(future_pred_head=dict(history_queue_length=0))
work_dir = 'work_dirs/farmsim_panoptic_front3_s'
