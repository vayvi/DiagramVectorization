import os
import torch
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import cv2
import json
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp_folder", type=str, default="res50_stage1_circles")
parser.add_argument("--epoch", type=str, default="400")
parser.add_argument("--real_data", action="store_true")
parser.add_argument(
    "--line_thresholds",
    nargs=3,
    type=int,
    default=[5, 10, 15],
    help="List of three integer thresholds for lines",
)
parser.add_argument(
    "--circle_thresholds",
    nargs=3,
    type=int,
    default=[5, 10, 15],
    help="List of three integer thresholds for circles",
)


def get_l2_distance(pred_circles, gt_circles):
    diff = ((pred_circles[:, None, :, None] - gt_circles[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        np.sqrt(diff[:, :, 0, 0] + diff[:, :, 1, 1]),
        np.sqrt(diff[:, :, 0, 1] + diff[:, :, 1, 0]),
    )
    return diff


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


def msTPFP(line_pred, line_gt, threshold, get_fn=False):
    if (len(line_gt) > 0) and (len(line_pred) > 0):
        diff = get_l2_distance(line_pred, line_gt)

        choice = np.argmin(diff, 1)
        dist = np.min(diff, 1)
        hit = np.zeros(len(line_gt), bool)
        tp = np.zeros(len(line_pred), float)
        fp = np.zeros(len(line_pred), float)

        for i in range(len(line_pred)):
            if dist[i] < threshold and not hit[choice[i]]:
                hit[choice[i]] = True
                tp[i] = 1
            else:
                fp[i] = 1
        fn = 1 - hit
    elif len(line_gt) == 0:
        tp = np.zeros(len(line_pred), float)
        fp = np.ones(len(line_pred))
        fn = np.zeros(len(line_gt), float)
    else:
        tp = np.zeros(len(line_pred), float)
        fp = np.zeros(len(line_pred), float)
        fn = np.ones(len(line_gt), float)
    if get_fn:
        return tp.sum(), fp.sum(), fn.sum()

    return tp, fp


def line_fscore(
    path,
    GT_val,
    line_threshold=5,
    circle_threshold=5,
):
    preds = sorted(glob.glob(path))
    print(len(preds))
    if "val" in str(path):
        gts = sorted(glob.glob(GT_val))
        print(len(gts))
    else:
        print(str(path))
        raise NotImplementedError("Only validation set is supported for now")
    tps, fps, fns = 0, 0, 0
    tps_circles, fps_circles, fns_circles = 0, 0, 0
    for pred_name, gt_name in zip(preds, gts):
        im_name = os.path.basename(pred_name).split(".")[0]
        gt_im_name = os.path.basename(gt_name).split(".")[0]

        assert (
            im_name in gt_im_name
        ), f"pred {im_name} and gt {gt_im_name} do not match for image"
        with np.load(pred_name) as fpred:
            primitives = fpred["shapes"][:, :, :2]
            mask = fpred["label"]
            pred_line = primitives[~mask.astype(bool)]  # 0 for line, 1 for circle
            line_score = fpred["score"][~mask.astype(bool)]
            pred_line = pred_line[line_score > 0.7]

            pred_circle = primitives[mask.astype(bool)]
            circle_score = fpred["score"][mask.astype(bool)]
            pred_circle = pred_circle[circle_score > 0.7]
        with np.load(gt_name) as fgt:
            try:
                gt_line = fgt["lines"][:, :, :2]
            except IndexError:
                print(f"##### {im_name} has no lines ")
                gt_line = []
            try:
                gt_circle = fgt["circles"][:, :, :2]
            except IndexError:
                print(f"##### {im_name} has no lines ")
                gt_circle = []

        tp, fp, fn = msTPFP(
            pred_line,
            gt_line,
            line_threshold,
            get_fn=True,
        )

        tp_circles, fp_circles, fn_circles = msTPFP(
            pred_circle,
            gt_circle,
            circle_threshold,
            get_fn=True,
        )
        print(
            f"## {im_name} ## Line metrics: {get_precision_recall_fscore(tp, fp, fn)}"
        )

        print(
            f"## {im_name} ## Circle metrics: {get_precision_recall_fscore(tp_circles, fp_circles, fn_circles)}"
        )
        tps += tp
        fps += fp
        fns += fn

        tps_circles += tp_circles
        fps_circles += fp_circles
        fns_circles += fn_circles

    precision, recall, fscore = get_precision_recall_fscore(tps, fps, fns)
    precision_circles, recall_circles, fscore_circles = get_precision_recall_fscore(
        tps_circles, fps_circles, fns_circles
    )

    print(
        f"""Final metrics are: \n Lines precision:{precision} recall:{recall} fscore:{fscore} 
        \n 
        Circles precision:{precision_circles} recall:{recall_circles} fscore :{fscore_circles} 
        """
    )
    # return {"line": [tp, fp, fn], "circle": [tp_circles, fp_circles, fn_circles]}


ROOT_DIR = Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    args = parser.parse_args()
    if args.real_data:
        exp_folder = ROOT_DIR / f"exp/{args.exp_folder}/real_val"
        GT_path = ROOT_DIR / "data/diagrams/valid_labels/*.npz"
    else:
        exp_folder = ROOT_DIR / f"exp/{args.exp_folder}/synthetic_val"
        GT_path = ROOT_DIR / "data/synthetic_raw/valid_labels/*.npz"

    pred_path = str((exp_folder / (f"epoch_{args.epoch}/*.npz")))
    print(pred_path)
    print(GT_path)
    eval_folder = exp_folder / f"evaluation/epoch_{args.epoch}"
    line_thresholds, circle_thresholds = args.line_thresholds, args.circle_thresholds
    for line_threshold, circle_threshold in zip(line_thresholds, circle_thresholds):
        line_fscore(
            pred_path,
            str(GT_path),
            line_threshold=line_threshold,
            circle_threshold=circle_threshold,
        )
        print("line_threshold", line_threshold)
        print("circle_threshold", circle_threshold)
