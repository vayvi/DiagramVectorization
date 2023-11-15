from ast import arg
import os
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
from ransac.parse_svg import BadPath, get_gt_from_svg
from ransac.evaluation import get_precision_recall_fscore, get_tp_fp_fn
from ransac.parse_pkl import parse_pkl
from ransac import OUTPUT_DATADIR, INPUT_DATADIR, DATADIR, max_dimension
from PIL import Image, ImageDraw


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_folder",
        type=str,
        default="ransac_wo_text_1500_7_300_1200_5_joint_no_outliers_resized",
        help="folder with predicted lines and circles.",
    )
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
    parser.add_argument(
        "--annotation_folder",
        type=str,
        default="svgs",
        help="folder with annotation svgs.",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Threshold for metrics.",
    )
    return parser


def save_gt(gt_dict, diagram_name):
    size = int(gt_dict["width"]), int(gt_dict["height"])
    canvas = Image.new("RGBA", size)
    draw = ImageDraw.Draw(canvas)
    for line in gt_dict["lines"]:
        draw.line([line[0, 0], line[0, 1], line[1, 0], line[1, 1]])
    for center, circle_radius in zip(
        gt_dict["circles"]["centers"], gt_dict["circles"]["radii"]
    ):
        shape = [
            center[0] - circle_radius,
            center[1] - circle_radius,
            center[0] + circle_radius,
            center[1] + circle_radius,
        ]
        draw.ellipse(shape)
    canvas.save((INPUT_DATADIR / "scaled_gt_plots") / f"{diagram_name}.png")


def scale_gt_dict(scale_ratio, gt_dict):
    for key in ["width", "height"]:
        gt_dict[key] = round(gt_dict[key] * scale_ratio)
    gt_dict["lines"] = gt_dict["lines"] * scale_ratio
    for key in ["centers", "radii"]:  # FIXME: ellipses are missing
        gt_dict["circles"][key] = gt_dict["circles"][key] * scale_ratio
    return gt_dict


def get_resized_dimensions(width, height, max_dimension=1000):
    if max(width, height) > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int((height / width) * max_dimension)
        else:
            new_width = int((width / height) * max_dimension)
            new_height = max_dimension
    return new_width, new_height


def read_pred_and_gt(diagram_name, pred_folder, annotation_folder, percentile=1):
    gt_dict = get_gt_from_svg(annotation_folder / f"{diagram_name}-corr.svg")
    width, height = gt_dict["width"], gt_dict["height"]
    if max(width, height) > max_dimension:
        width, height = get_resized_dimensions(width, height)
    pred_lines, pred_circles = parse_pkl(
        pred_folder / diagram_name, width, height, percentile
    )
    return np.array(pred_lines), pred_circles, gt_dict


def eval(
    diagram_name,
    pred_folder,
    annotation_folder,
    line_threshold,
    circle_threshold=10,
    percentile=1,
    new_height=None,
):
    if not os.path.exists(pred_folder / f"{diagram_name}"):
        raise FileNotFoundError(
            f"{diagram_name}.png doesn't exist. Check for . or space in name"
        )

    pred_lines, pred_circles, gt_dict = read_pred_and_gt(
        diagram_name, pred_folder, annotation_folder, percentile
    )
    # print(pred_circles)

    w, h = gt_dict["width"], gt_dict["height"]

    if min(w, h) > max_dimension:
        if new_height is None:
            new_width, new_height = get_resized_dimensions(w, h)
        scale_ratio = new_height / gt_dict["height"]
        print(f"## Rescaling {diagram_name} with {scale_ratio} ##")
        gt_dict = scale_gt_dict(scale_ratio, gt_dict)
        # save_gt(gt_dict, diagram_name)
        # print(pred_lines, gt_dict["lines"])
        # print(pred_circles, gt_dict["circles"])

    # print("###########################")
    # print(pred_lines, gt_dict["lines"])
    # print("###########################")

    # print(f"Diagram {diagram_name} has curves inside the path")
    tp, fp, fn = get_tp_fp_fn(
        pred_lines, gt_dict["lines"], primitive="line", threshold=line_threshold
    )
    print(
        f"## {diagram_name} ## Line metrics: {get_precision_recall_fscore(tp, fp, fn)}"
    )

    tp_c, fp_c, fn_c = get_tp_fp_fn(
        pred_circles, gt_dict["circles"], primitive="circle", threshold=circle_threshold
    )
    print(
        f"## {diagram_name} ## Circle metrics: {get_precision_recall_fscore(tp_c, fp_c, fn_c)}"
    )
    return {"line": [tp, fp, fn], "circle": [tp_c, fp_c, fn_c]}


def main(
    pred_folder,
    annotation_folder,
    line_threshold,
    circle_threshold,
    rescale_dict=None,
    percentile=5,
):
    tps, fps, fns = 0, 0, 0
    tps_circles, fps_circles, fns_circles = 0, 0, 0
    diagram_names = []

    print(annotation_folder)

    # for e in annotation_folder.glob("-corr.svg"):
    #     print(e)
    #     print("#########")
    #     break

    for gt_path in tqdm(annotation_folder.glob("*corr*.svg")):
        diagram_name = os.path.basename(gt_path)[:-9]  # removing -corr.svg

        print(f"##########  {diagram_name} ##########")
        if rescale_dict is not None:
            new_height = rescale_dict[diagram_name]
        else:
            new_height = None
        try:
            eval_dict = eval(
                diagram_name,
                pred_folder,
                annotation_folder,
                line_threshold,
                circle_threshold,
                percentile=percentile,
                new_height=new_height,
            )
        except BadPath:
            print(f"Diagram {diagram_name} has curves inside the path")
            continue
        except FileNotFoundError as e:
            print(e)
            continue

        tp, fp, fn = eval_dict["line"]
        tps += tp
        fps += fp
        fns += fn

        tp_circles, fp_circles, fn_circles = eval_dict["circle"]
        tps_circles += tp_circles
        fps_circles += fp_circles
        fns_circles += fn_circles

        diagram_names.append(diagram_name)

    precision, recall, fscore = get_precision_recall_fscore(tps, fps, fns)
    precision_circles, recall_circles, fscore_circles = get_precision_recall_fscore(
        tps_circles, fps_circles, fns_circles
    )
    print(
        f"""Threshold {line_threshold} \n Lines precision:{precision} recall:{recall} fscore:{fscore} 
          \n 
          Threshold {circle_threshold} Circles precision:{precision_circles} recall:{recall_circles} fscore :{fscore_circles} 
          """
    )

    # print(diagram_names, len(diagram_names))
    dir_path, folder_name = os.path.split(pred_folder)
    json_file_path = os.path.join(dir_path, "evaluation_results.json")
    results_dict = {
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "precision_c": precision_circles,
        "recall_c": recall_circles,
        "fscore_c": fscore_circles,
        "fscore_sum": fscore + fscore_circles,
    }
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            data = json.load(file)
        data.append(results_dict)
        with open(json_file_path, "w") as file:
            json.dump(data, file, indent=4)
    else:
        print(f"writing results to {json_file_path}")
        with open(json_file_path, "w") as file:
            json.dump([results_dict], file, indent=4)

    return precision, recall, fscore, precision_circles, recall_circles, fscore_circles


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pred_folder = (OUTPUT_DATADIR / args.pred_folder) / "info"
    annotation_folder = INPUT_DATADIR / args.annotation_folder
    for line_threshold, circle_threshold in zip(
        args.line_thresholds, args.circle_thresholds
    ):
        main(pred_folder, annotation_folder, line_threshold, circle_threshold)
