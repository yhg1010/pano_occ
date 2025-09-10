# Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
# Modified by Haisong Liu
import math
import copy
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tqdm import tqdm
from prettytable import PrettyTable
from .ray_pq import Metric_RayPQ
from .old_metrics import Metric_Panoptic, Metric_mIoU

# dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])
_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4


# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/test_fgbg.py#L29C1-L46C16
def get_rendered_pcds(origin, points, tindex, pred_dist):
    pcds = []
    
    for t in range(len(origin)):
        mask = (tindex == t)
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))
        
    return pcds


def meshgrid3d(occ_size, pc_range):
    W, H, D = occ_size
    
    xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
    zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
    xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
    ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
    zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack((xs, ys, zs), -1)

    return xyz


def generate_lidar_rays():
    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)
    
    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)


def process_one_sample(sem_pred, lidar_rays, output_origin, instance_pred=None, occ_class_names=None):
    # lidar origin in ego coordinate
    # lidar_origin = torch.tensor([[[0.9858, 0.0000, 1.8402]]])
    T = output_origin.shape[1]
    pred_pcds_t = []

    free_id = len(occ_class_names) - 1 
    occ_pred = copy.deepcopy(sem_pred)
    occ_pred[sem_pred < free_id] = 1
    occ_pred[sem_pred == free_id] = 0
    occ_pred = occ_pred.permute(2, 1, 0)
    occ_pred = occ_pred[None, None, :].contiguous().float()

    offset = torch.Tensor(_pc_range[:3])[None, None, :]
    scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]

    lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])
    
    for t in range(T): 
        lidar_origin = output_origin[:, t:t+1, :]  # [1, 1, 3]
        lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

        output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
        output_tindex_render = lidar_tindex  # [1, N], all zeros

        with torch.no_grad():
            pred_dist, _, coord_index = dvr.render_forward(
                occ_pred.cuda(),
                output_origin_render.cuda(),
                output_points_render.cuda(),
                output_tindex_render.cuda(),
                [1, 16, 200, 200],
                "test"
            )
            pred_dist *= _voxel_size

        pred_pcds = get_rendered_pcds(
            lidar_origin[0].cpu().numpy(),
            lidar_endpts[0].cpu().numpy(),
            lidar_tindex[0].cpu().numpy(),
            pred_dist[0].cpu().numpy()
        )
        coord_index = coord_index[0, :, :].int().cpu()  # [N, 3]

        pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]  # [N, 1]        
        pred_dist = pred_dist[0, :, None].cpu()

        if instance_pred is not None:
            pred_instance = instance_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]  # [N, 1]
            pred_pcds = torch.cat([pred_label.float(), pred_instance.float(), pred_dist], dim=-1)
        else:
            pred_pcds = torch.cat([pred_label.float(), pred_dist], dim=-1)

        pred_pcds_t.append(pred_pcds)

    pred_pcds_t = torch.cat(pred_pcds_t, dim=0)
   
    return pred_pcds_t.numpy()


def calc_rayiou(pcd_pred_list, pcd_gt_list, occ_class_names):
    thresholds = [1, 2, 4]

    gt_cnt = np.zeros([len(occ_class_names)])
    pred_cnt = np.zeros([len(occ_class_names)])
    tp_cnt = np.zeros([len(thresholds), len(occ_class_names)])

    for pcd_pred, pcd_gt in zip(pcd_pred_list, pcd_gt_list):
        for j, threshold in enumerate(thresholds):
            # L1
            depth_pred = pcd_pred[:, 1]
            depth_gt = pcd_gt[:, 1]
            l1_error = np.abs(depth_pred - depth_gt)
            tp_dist_mask = (l1_error < threshold)
            
            for i, cls in enumerate(occ_class_names):
                cls_id = occ_class_names.index(cls)
                cls_mask_pred = (pcd_pred[:, 0] == cls_id)
                cls_mask_gt = (pcd_gt[:, 0] == cls_id)

                gt_cnt_i = cls_mask_gt.sum()
                pred_cnt_i = cls_mask_pred.sum()
                if j == 0:
                    gt_cnt[i] += gt_cnt_i
                    pred_cnt[i] += pred_cnt_i

                tp_cls = cls_mask_gt & cls_mask_pred  # [N]
                tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                tp_cnt[j][i] += tp_mask.sum()
    
    iou_list = []
    for j, threshold in enumerate(thresholds):
        iou_list.append((tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j]))[:-1])

    return iou_list


def main_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list, occ_class_names):
    torch.cuda.empty_cache()

    # generate lidar rays
    lidar_rays = generate_lidar_rays()
    lidar_rays = torch.from_numpy(lidar_rays)

    pcd_pred_list, pcd_gt_list = [], []
    for sem_pred, sem_gt, lidar_origins in tqdm(zip(sem_pred_list, sem_gt_list, lidar_origin_list), ncols=50):
        sem_pred = torch.from_numpy(np.reshape(sem_pred, [200, 200, 16]))
        sem_gt = torch.from_numpy(np.reshape(sem_gt, [200, 200, 16]))

        pcd_pred = process_one_sample(sem_pred, lidar_rays, lidar_origins, occ_class_names=occ_class_names)
        pcd_gt = process_one_sample(sem_gt, lidar_rays, lidar_origins, occ_class_names=occ_class_names)

        # evalute on non-free rays
        valid_mask = (pcd_gt[:, 0].astype(np.int32) != len(occ_class_names) - 1)
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        pcd_pred_list.append(pcd_pred)
        pcd_gt_list.append(pcd_gt)

    iou_list = calc_rayiou(pcd_pred_list, pcd_gt_list, occ_class_names)
    rayiou = np.nanmean(iou_list)
    rayiou_0 = np.nanmean(iou_list[0])
    rayiou_1 = np.nanmean(iou_list[1])
    rayiou_2 = np.nanmean(iou_list[2])
    
    table = PrettyTable([
        'Class Names',
        'RayIoU@1', 'RayIoU@2', 'RayIoU@4'
    ])
    table.float_format = '.3'

    for i in range(len(occ_class_names) - 1):
        table.add_row([
            occ_class_names[i],
            iou_list[0][i], iou_list[1][i], iou_list[2][i]
        ], divider=(i == len(occ_class_names) - 2))
    
    table.add_row(['MEAN', rayiou_0, rayiou_1, rayiou_2])

    print(table)

    torch.cuda.empty_cache()

    return {
        'RayIoU': rayiou,
        'RayIoU@1': rayiou_0,
        'RayIoU@2': rayiou_1,
        'RayIoU@4': rayiou_2,
    }


def main_raypq(sem_pred_list, sem_gt_list, inst_pred_list, inst_gt_list, lidar_origin_list, occ_class_names):
    torch.cuda.empty_cache()

    eval_metrics_pq = Metric_RayPQ(
        occ_class_names=occ_class_names,
        num_classes=len(occ_class_names),
        thresholds=[1, 2, 4]
    )

    # generate lidar rays
    lidar_rays = generate_lidar_rays()
    lidar_rays = torch.from_numpy(lidar_rays)

    for sem_pred, sem_gt, inst_pred, inst_gt, lidar_origins in \
        tqdm(zip(sem_pred_list, sem_gt_list, inst_pred_list, inst_gt_list, lidar_origin_list), ncols=50):
        sem_pred = torch.from_numpy(np.reshape(sem_pred, [200, 200, 16]))
        sem_gt = torch.from_numpy(np.reshape(sem_gt, [200, 200, 16]))

        inst_pred = torch.from_numpy(np.reshape(inst_pred, [200, 200, 16]))
        inst_gt = torch.from_numpy(np.reshape(inst_gt, [200, 200, 16]))

        pcd_pred = process_one_sample(sem_pred, lidar_rays, lidar_origins, instance_pred=inst_pred, occ_class_names=occ_class_names)
        pcd_gt = process_one_sample(sem_gt, lidar_rays, lidar_origins, instance_pred=inst_gt, occ_class_names=occ_class_names)

        # evalute on non-free rays
        valid_mask = (pcd_gt[:, 0].astype(np.int32) != len(occ_class_names) - 1)
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        
        sem_gt = pcd_gt[:, 0].astype(np.int32)
        sem_pred = pcd_pred[:, 0].astype(np.int32)

        instances_gt = pcd_gt[:, 1].astype(np.int32)
        instances_pred = pcd_pred[:, 1].astype(np.int32)

        # L1
        depth_gt = pcd_gt[:, 2]
        depth_pred = pcd_pred[:, 2]
        l1_error = np.abs(depth_pred - depth_gt)

        eval_metrics_pq.add_batch(sem_pred, sem_gt, instances_pred, instances_gt, l1_error)

    torch.cuda.empty_cache()

    return eval_metrics_pq.count_pq()


def main_pq(sem_pred_list, sem_gt_list, inst_pred_list, inst_gt_list):
    eval_metrics_pq = Metric_Panoptic()

    for sem_pred, sem_gt, inst_pred, inst_gt in \
        tqdm(zip(sem_pred_list, sem_gt_list, inst_pred_list, inst_gt_list), ncols=50):
        # inst_gt = inst_gt.astype(np.int)
        eval_metrics_pq.add_batch(sem_pred, sem_gt, inst_pred, inst_gt)
    
    return eval_metrics_pq.count_pq()

def main_iou(sem_pred_list, sem_gt_list, ignore_masks):
    eval_miou = Metric_mIoU()

    for sem_pred, sem_gt, ignore_mask in tqdm(zip(sem_pred_list, sem_gt_list, ignore_masks), ncols=50):
        valid_mask = sem_gt != 17
        sem_pred = sem_pred[valid_mask]
        sem_gt = sem_gt[valid_mask]
        ignore_mask = ignore_mask[valid_mask]
        eval_miou.add_batch(sem_pred, sem_gt.numpy(), ignore_mask)
    
    return eval_miou.count_miou()


def main_iou2(sem_pred_list, sem_gt_list):
    results = []
    for i in range(len(sem_pred_list)):
        gt_i, pred_i = sem_gt_list[i].numpy(), sem_pred_list[i]
        # gt_i = gt_to_voxel(gt_i, img_metas)
        gt_i = gt_i.squeeze()
        mask = (gt_i != 17)
        score = np.zeros((18, 3))
        for j in range(18):
            if j == 17: #class 0 for geometry IoU
                score[j][0] += ((gt_i!=17) * (pred_i!=17)).sum()
                score[j][1] += (gt_i!=17).sum()
                score[j][2] += (pred_i!=17).sum()
            else:
                score[j][0] += ((gt_i == j) * (pred_i == j)).sum()
                score[j][1] += (gt_i == j).sum()
                score[j][2] += (pred_i == j).sum()

        results.append(score)
    
    results = np.stack(results, axis=0)
    import ipdb; ipdb.set_trace()
    class_names = {17: 'IoU'}
    class_num = 18
    occ_class_names = [
        'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk',
        'terrain', 'manmade', 'vegetation'
    ]
    for i, name in enumerate(occ_class_names):
        class_names[i] = occ_class_names[i]
    
    results = np.stack(results, axis=0).mean(0)
    mean_ious = []
    results_dict = {}
    import ipdb; ipdb.set_trace()
    for i in range(class_num):
        tp = results[i, 0]
        p = results[i, 1]
        g = results[i, 2]
        union = p + g - tp
        mean_ious.append(tp / union)
    
    for i in range(class_num):
        results_dict[class_names[i]] = mean_ious[i]
    import ipdb; ipdb.set_trace()
    results_dict['mIoU'] = np.nanmean(np.array(mean_ious)[0:-1])
    print(results_dict)



def convert_to_voxel_list(pano_sem_results, pano_inst_results, voxel_semantics, voxel_instances, ignore_masks=None):
    batch_size = len(pano_sem_results)
    results = []
    for i in range(batch_size):
        pano_sem = pano_sem_results[i][ignore_masks[i]].reshape(-1)
        pano_inst = pano_inst_results[i][ignore_masks[i]].reshape(-1)
        voxel_sem = voxel_semantics[i][ignore_masks[i]].reshape(-1).numpy()
        voxel_inst = voxel_instances[i][ignore_masks[i]].reshape(-1).numpy()

        # # Filter out empty voxels
        # valid_indices = pano_sem != 17
        # pano_sem = pano_sem[valid_indices]
        # pano_inst = pano_inst[valid_indices]
        # import ipdb; ipdb.set_trace()
        # valid_indices = voxel_sem != 17
        # voxel_sem = voxel_sem[valid_indices]
        # voxel_inst = voxel_inst[valid_indices]

        # # Filter out instance id 255 in ground truth
        # valid_indices = voxel_inst != 255
        # voxel_sem = voxel_sem[valid_indices]
        # voxel_inst = voxel_inst[valid_indices]

        results.append((pano_sem, pano_inst, voxel_sem, voxel_inst))

    return results

def compute_pq_single(pano_sem, pano_inst, voxel_sem, voxel_inst):
    classes_pred = np.unique(pano_sem)
    num_classes = len(classes_pred)
    # import ipdb; ipdb.set_trace()
    tp = np.zeros(17)
    fp = np.zeros(17)
    fn = np.zeros(17)
    iou_sum = np.zeros(17)

    for gt_id in np.unique(voxel_inst):
        if gt_id == 255:
            continue
        gt_mask = voxel_inst == gt_id
        gt_class = np.unique(voxel_sem[gt_mask])
        # import ipdb; ipdb.set_trace()
        if len(gt_class) != 1:
            continue
        gt_class = gt_class[0]

        matched_pred_idx = -1
        max_iou = 0
        for pred_id in np.unique(pano_inst):
            pred_mask = pano_inst == pred_id
            pred_class = np.unique(pano_sem[pred_mask])
            if len(pred_class) != 1 or pred_class[0] != gt_class:
                continue

            pred_class = pred_class[0]
            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)
            iou = intersection / union if union > 0 else 0

            if iou > max_iou:
                max_iou = iou
                matched_pred_idx = pred_id

        if max_iou > 0.:
            tp[gt_class] += 1
            iou_sum[gt_class] += max_iou
        else:
            fn[gt_class] += 1

    for pred_id in np.unique(pano_inst):
        pred_mask = pano_inst == pred_id
        pred_class = np.unique(pano_sem[pred_mask])
        if len(pred_class) != 1:
            continue
        pred_class = pred_class[0]
        if not any(np.sum(pred_mask & (voxel_inst == gt_id)) > 0 for gt_id in np.unique(voxel_inst)):
            fp[pred_class] += 1

    pq = np.zeros(17)
    for c in range(17):
        if tp[c] + fp[c] + fn[c] > 0:
            pq[c] = iou_sum[c] / (tp[c] + 0.5 * fp[c] + 0.5 * fn[c])

    pq_mean = np.mean(pq)
    # import ipdb; ipdb.set_trace()
    return pq_mean, pq

from tqdm import tqdm

def compute_pq_batch(pano_sem_results, voxel_semantics, pano_inst_results, voxel_instances, ignore_masks=None):
    converted_data = convert_to_voxel_list(pano_sem_results, pano_inst_results, voxel_semantics, voxel_instances, ignore_masks)
    total_pq = np.zeros(17)  # assuming 18 classes including background
    count = np.zeros(17)
    # import ipdb; ipdb.set_trace()
    for pano_sem, pano_inst, voxel_sem, voxel_inst in tqdm(converted_data):
        pq_mean, pq_per_class = compute_pq_single(pano_sem, pano_inst, voxel_sem, voxel_inst)
        total_pq += pq_per_class
        count += (pq_per_class > 0)

    avg_pq_per_class = total_pq / np.maximum(count, 1)
    avg_pq = np.mean(avg_pq_per_class)
    return {
        'PQ': avg_pq,
        'PQ per class': avg_pq_per_class
    }
    # return avg_pq, avg_pq_per_class