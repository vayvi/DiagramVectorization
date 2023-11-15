import torch
import yaml
import glob
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
from letr_extension.postprocess import process_image, load_model, dict_to_namespace
import json
import numpy as np
import os
import argparse


root = Path(__file__).resolve().parent

exp_folder = root / "exp/res50_stage2_good_crops"


def msTPFP(line_pred, line_gt, threshold):
    if len(line_gt) > 0:
        diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)

        diff = np.minimum(
            diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
        )
        choice = np.argmin(diff, 1)

        dist = np.min(diff, 1)
        hit = np.zeros(len(line_gt), bool)
        tp = np.zeros(len(line_pred), int)
        tp_dist = np.zeros(len(line_pred), float)
        fp = np.zeros(len(line_pred), int)
        for i in range(len(line_pred)):
            if dist[i] < threshold and not hit[choice[i]]:
                hit[choice[i]] = True
                tp[i] = 1
                tp_dist[i] = dist[i]
            else:
                fp[i] = 1
    else:
        tp = np.zeros(len(line_pred), int)
        fp = np.ones(len(line_pred), int)
    return tp, fp, tp_dist, choice


def plot_results(
    pil_img,
    prob,
    boxes,
    gt_boxes,
    thresh_line,
    thresh_circle,
    plot_proba=False,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 30))
    ax1.imshow(pil_img, aspect="equal")
    plt.axis("off")
    ax2.imshow(pil_img, aspect="equal")
    plt.axis("off")
    colors = plt.cm.hsv(np.linspace(0, 1, 30)).tolist()
    k = 0
    for p, line, gt_line, color in zip(prob, boxes, gt_boxes, colors):
        fig = plt.figure(figsize=(10, 10))
        ax2 = fig.add_subplot(111)
        ax2.imshow(pil_img, aspect="equal")
        xmin, ymin, xmax, ymax = line
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt_line

        cl = p.argmax()
        if cl == 0:
            if p.max() > thresh_line:
                ax2.plot(
                    [xmin, xmax],
                    [ymin, ymax],
                    linewidth=1,
                    color=color,
                    zorder=1,
                )
                ax2.plot(
                    [xmin_gt, xmax_gt],
                    [ymin_gt, ymax_gt],
                    linewidth=1,
                    color=color,
                    zorder=1,
                    linestyle="--",
                )
                ax2.text(
                    xmin_gt,
                    ymin_gt,
                    "gt",
                    fontsize=15,
                    bbox=dict(facecolor="yellow", alpha=0.1),
                )
                text = f"l: {p[cl]:0.2f}"
                if plot_proba:
                    ax2.text(
                        xmin,
                        ymin,
                        text,
                        fontsize=15,
                        bbox=dict(facecolor="yellow", alpha=0.1),
                    )
        else:
            if p.max() > thresh_circle:
                r1 = (xmax - xmin) / 2
                r2 = (ymax - ymin) / 2
                r1_gt = (xmax_gt - xmin_gt) / 2
                r2_gt = (ymax_gt - ymin_gt) / 2
                if (r1 > 0) and (r2 > 0):
                    center = (xmin + r1, ymin + r2)
                    center_gt = (xmin_gt + r1_gt, ymin_gt + r2_gt)
                    ax2.add_patch(
                        plt.Circle(center, r2, color=color, fill=False, linewidth=1)
                    )
                    ax2.add_patch(
                        plt.Circle(
                            center_gt,
                            r2_gt,
                            color=color,
                            fill=False,
                            linewidth=1,
                            linestyle="--",
                        )
                    )
                    ax2.text(
                        center_gt[0],
                        center_gt[1] + r1_gt,
                        "gt",
                        fontsize=15,
                        bbox=dict(facecolor="yellow", alpha=0.1),
                    )
                    if plot_proba:
                        ax2.text(
                            center[0],
                            center[1] + r1,
                            f"p={p[cl]:.2f}",
                            fontsize=15,
                            bbox=dict(facecolor="yellow", alpha=0.1),
                        )
                else:
                    print("r1 or r2 is negative", r1, r2)
        fig.savefig(f"exp/res50_stage2_good_crops/results/test_{k}.png")
        k = k + 1
    exit()
    plt.axis("off")
    plt.show()
    return fig


def get_gt_annotation(im_data):
    size = (im_data["width"], im_data["height"])
    center = np.array(im_data["circle_centers"])
    radii = np.array(im_data["circle_radii"])
    xs = np.column_stack((center[:, 0] - radii, center[:, 0] + radii))
    ys = np.column_stack((center[:, 1] - radii, center[:, 1] + radii))

    circles_absolute = np.dstack((xs, ys))
    circles_relative = np.dstack((xs / size[0], ys / size[1]))

    lines_absolute = np.array(im_data["lines"])
    lines_relative = lines_absolute / np.array([*size, *size])

    lines_absolute = lines_absolute.reshape(-1, 2, 2)
    lines_relative = lines_relative.reshape(-1, 2, 2)

    return lines_relative, circles_relative, lines_absolute, circles_absolute


def main(args):
    with open(args.config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = dict_to_namespace(config_dict)

    model = load_model(cfg, exp_folder / "checkpoints")
    model.eval()

    folder_path = root / args.input_dir

    gt_annotation_path = folder_path.parent / "valid.json"

    with open(gt_annotation_path, "r") as f:
        gt_data = json.load(f)

    for im_data in gt_data:
        im_name = im_data["filename"]
        (
            lines_relative,
            circles_relative,
            lines_absolute,
            circles_absolute,
        ) = get_gt_annotation(im_data)

        print("currently processing image", im_name)
        im_path = folder_path / im_name
        # im_name = os.path.basename(im_path)

        if os.path.exists((exp_folder / "results") / im_name):
            print("Image prediction already exists", im_name)
            continue
        else:
            print("Predicting image", im_name)

        try:
            raw_img = plt.imread(im_path)[:, :, :3]
        except IndexError as e:
            print(e)
            continue

        inputs, orig_size = process_image(raw_img)
        outputs = model(inputs)["shapes"]  # FIXME: make this more flexible
        out_logits, out_line = outputs["pred_logits"], outputs["pred_shapes"]
        prob = F.softmax(out_logits, -1)[0, :, :-1]
        threshold = 0.5
        keep = prob.max(-1).values > threshold
        prob = prob[keep]
        out_line = out_line[0, keep]
        img_h, img_w = orig_size.unbind(0)
        scale_fct = torch.unsqueeze(
            torch.stack([img_w, img_h, img_w, img_h], dim=0), dim=0
        )

        primitives_relative = out_line.view(len(out_line), 2, 2).detach().numpy()

        mask = prob[:, 0] > prob[:, 1]

        tp, fp, tp_dist, choice_lines = msTPFP(
            primitives_relative[mask], lines_relative, 0.1
        )
        print("TP", tp, "FP", fp, tp_dist)
        tp_c, fp_c, tp_dist_c, choice_circles = msTPFP(
            primitives_relative[~mask], circles_relative, 0.05
        )
        print("TP_circles", tp_c, "FP_circles", fp_c, tp_dist_c)

        primitives_absolute = out_line * scale_fct[:, None, :]
        primitives_absolute = (
            primitives_absolute.view(len(out_line), 2, 2).detach().numpy()
        )
        gt_boxes = np.zeros_like(primitives_absolute)
        tp, tp_c = np.array(tp, bool), np.array(tp_c, bool)

        gt_lines, gt_circles = gt_boxes[mask], gt_boxes[~mask]
        gt_lines[tp] = lines_absolute[choice_lines][tp]
        gt_circles[tp_c] = circles_absolute[choice_circles][tp_c]
        gt_boxes[mask] = gt_lines
        gt_boxes[~mask] = gt_circles
        primitives_absolute = primitives_absolute.reshape(
            primitives_absolute.shape[0], -1
        )

        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], -1)
        fig = plot_results(
            raw_img,
            prob,
            primitives_absolute,
            gt_boxes,
            thresh_line=0.7,
            thresh_circle=0.7,
            plot_proba=True,
        )
        break
        fig.savefig((exp_folder / "results") / im_name, bbox_inches="tight")
        plt.close(fig)
        break


if __name__ == "__main__":
    os.makedirs(exp_folder / "results", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config_primitives.yaml")
    parser.add_argument("--input_path", type=str, default="data/diagrams/images")
    args = parser.parse_args()
    main(args)
