_base_ = ['./r50_nuimg_704x256_8f.py']

# occ_gt_root = 'data/nuscenes/occ3d_panoptic'
occ_gt_root = 'data/nuscenes/occ3d_pano'

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]

# For nuScenes we usually do 10-class detection
det_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]


_num_frames_ = 8


model = dict(
    use_grid_mask=False,
    use_mask_camera=False,
    pts_bbox_head=dict(
        panoptic=True
    )
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': False,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names), inst_class_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],  # other keys: 'mask_camera'
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names), inst_class_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]

data = dict(
    workers_per_gpu=8,
    train=dict(
        pipeline=train_pipeline,
        occ_gt_root=occ_gt_root,
        ann_file='data/nuscenes/nuscenes_infos_train_sweep_v3.pkl'
    ),
    val=dict(
        pipeline=test_pipeline,
        occ_gt_root=occ_gt_root,
        ann_file='data/nuscenes/nuscenes_infos_val_sweep_v3.pkl'
    ),
    test=dict(
        pipeline=test_pipeline,
        occ_gt_root=occ_gt_root,
        ann_file='data/nuscenes/nuscenes_infos_val_sweep_v3.pkl'
    ),
)

total_epochs = 24
batch_size = 1

load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

resume_from = None
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='MyTextLoggerHook', interval=50, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ]
)

# evaluation
eval_config = dict(interval=total_epochs)