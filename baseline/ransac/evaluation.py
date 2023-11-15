import numpy as np


def get_distance_from_endpoints_to_line(pred_lines, gt_lines):
    gt_directions = (
        gt_lines[:, 1, :]
        - gt_lines[
            :,
            0,
        ]
    )
    gt_directions = gt_directions / np.linalg.norm(gt_directions)
    shifted_pred_directions = pred_lines[:, None, :, :] - gt_lines[:, None, 0, :]
    distances_endpoint1 = np.abs(
        np.cross(shifted_pred_directions[:, :, 0, :], gt_directions)
    )
    distances_endpoint2 = np.abs(
        np.cross(shifted_pred_directions[:, :, 1, :], gt_directions)
    )
    diff = distances_endpoint1 + distances_endpoint2

    return diff


def get_l2_distance(pred_lines, gt_lines):
    diff = ((pred_lines[:, None, :, None] - gt_lines[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    return diff


def get_distance_from_circles(pred_circles, gt_circles):
    pred_centers, gt_centers = pred_circles["centers"], gt_circles["centers"]
    pred_radii, gt_radii = pred_circles["radii"], gt_circles["radii"]
    center_distances = np.sqrt(
        ((pred_centers[:, None, :] - gt_centers[None, :]) ** 2).sum(-1)
    )
    radii_distances = np.abs(pred_radii[:, None] - gt_radii[None, :])
    diff = center_distances + radii_distances
    return diff


def get_tp_fp_fn(preds, gt, primitive="line", threshold=10):
    if primitive == "line":
        if len(preds) == 0:
            return 0, 0, len(gt)
        elif len(gt) == 0:
            return 0, len(preds), 0
        else:
            diff = get_distance_from_endpoints_to_line(preds, gt)
    elif primitive == "circle":
        if len(preds["centers"]) == 0:
            return 0, 0, len(gt["centers"])
        elif len(gt["centers"]) == 0:
            return 0, len(preds["centers"]), 0
        else:
            diff = get_distance_from_circles(preds, gt)
    else:
        raise ValueError(
            f"primitive {primitive} is unsupported. Valid choices are ['line', 'circle']"
        )

    nbr_preds, nbr_gt = diff.shape[0], diff.shape[1]
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(nbr_gt, bool)  #  ommiting already assigned ground truth.
    tp = np.zeros(nbr_preds, int)
    fp = np.zeros(nbr_preds, int)
    for i in range(nbr_preds):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    fn = 1 - hit  # can be deduce from tp but outputting it nonetheless
    return tp.sum(), fp.sum(), fn.sum()


def get_precision_recall_fscore(tp, fp, fn):
    if tp == 0:
        if fn == 0:
            return 0, 0, 1
        else:
            return 0, 0, 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore
