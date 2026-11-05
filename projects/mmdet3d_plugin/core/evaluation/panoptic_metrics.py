"""Voxel PQ over valid GT domain; free/void excluded, stuff merged per class."""
import numpy as np


def panoptic_counts(pred_sem, pred_inst, gt_sem, gt_inst,
                    thing_ids=(4, 5, 7, 8, 9, 10), num_classes=11):
    counts = np.zeros((num_classes, 4), dtype=np.float64)  # IoU sum, TP, FP, FN
    valid = gt_sem != 255
    ps, pi, gs, gi = (np.asarray(a)[valid] for a in (pred_sem, pred_inst, gt_sem, gt_inst))
    for c in range(1, num_classes):
        pm, gm = ps == c, gs == c
        # A rejected thing (ID0) is a prediction segment, never silently discarded.
        pids, pn = np.unique(pi[pm] if c in thing_ids else np.zeros(pm.sum(), dtype=int), return_counts=True)
        gids, gn = np.unique(gi[gm] if c in thing_ids else np.zeros(gm.sum(), dtype=int), return_counts=True)
        if not len(pids) or not len(gids):
            counts[c, 2:] = len(pids), len(gids)
            continue
        overlap = pm & gm
        p_overlap = pi[overlap] if c in thing_ids else np.zeros(overlap.sum(), dtype=int)
        g_overlap = gi[overlap] if c in thing_ids else np.zeros(overlap.sum(), dtype=int)
        pair = np.searchsorted(pids, p_overlap) * len(gids) + np.searchsorted(gids, g_overlap)
        keys, intersections = np.unique(pair, return_counts=True)
        pp, gg = keys // len(gids), keys % len(gids)
        ious = intersections / (pn[pp] + gn[gg] - intersections)
        matched = ious > .5
        if c in thing_ids:
            matched &= pids[pp] != 0  # Unassigned thing voxels cannot become a TP.
        tp = int(matched.sum())
        counts[c] = ious[matched].sum(), tp, len(pids) - tp, len(gids) - tp
    return counts


def panoptic_summary(value, class_names=None, thing_ids=(4, 5, 7, 8, 9, 10)):
    stats = np.asarray(value, dtype=np.float64)
    stats = stats.reshape(-1, stats.shape[-2], 4).sum(0)
    iou, tp, fp, fn = stats.T
    denominator = tp + .5 * (fp + fn)
    present = denominator > 0
    present[0] = False
    pq = np.divide(iou, denominator, out=np.zeros_like(iou), where=denominator > 0)
    sq = np.divide(iou, tp, out=np.zeros_like(iou), where=tp > 0)
    rq = np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    result = {}
    things = np.isin(np.arange(len(tp)), thing_ids)
    for suffix, mask in [('', present), ('_th', present & things), ('_st', present & ~things)]:
        for name, values in [('PQ', pq), ('SQ', sq), ('RQ', rq)]:
            result['panoptic_' + name + suffix] = float(values[mask].mean()) if mask.any() else 0.
    for c in range(1, len(tp)):
        name = class_names[c] if class_names is not None else str(c)
        result['panoptic_PQ_' + name] = float(pq[c])
        result['panoptic_support_' + name] = float(tp[c] + fn[c])
    return result
