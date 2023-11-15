
import cv2
import os
import argparse
import json
from ransac import INPUT_DATADIR, DATADIR, max_dimension, OUTPUT_DATADIR
from ransac import OUTPUT_DATADIR, ransac_line, ransac_circle
from ransac.utils import save_info, get_images
from pathlib import Path
from scripts.evaluate import main as main_eval
from collections import defaultdict
import yaml
import svgwrite


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        default="images_resized",
        help="folder where images are located.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="exp",
        help="folder where images will be saved",
    )

    parser.add_argument(
        "--output_info_folder",
        type=str,
        default="info",
        help="folder where detection information will be saved.",
    )
    parser.add_argument(
        "--edge_folder",
        type=str,
        default="edges_resized_testr_3",
        help="folder where edge images exist",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Maximal number of iterations (and detections)",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=1000,
        help="Maximal number trials in 1 ransac",
    )
    parser.add_argument(
        "--max_distance",
        type=int,
        default=1,
        help="residual distance threshold",
    )
    parser.add_argument(
        "--min_inliers_lines",
        type=int,
        default=300,
        help="residual distance threshold",
    )
    parser.add_argument(
        "--min_inliers_circles",
        type=int,
        default=500,
        help="residual distance threshold",
    )
    parser.add_argument(
        "--discard_large_circles",
        action="store_true",
        help="Discard circles with radii too large",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save plots in exp_folder/plots/percentile_k",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=1,
        help="line segment percentile to plot, plots from the kth to the (100-k)th percentile of the distribution.",
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
        "--joint",
        action="store_true",
        help="Lines and circles are detected jointly and not sequentially",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate over validation set after generating predictions",
    )
    parser.add_argument(
        "--resized",
        action="store_true",
        help="use resized image to have at most 1000 pixels in width or height",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config_baseline.yaml",
    )
    return parser


def detect_line(starting_points, line_results, args):
    inliers, outliers, model = ransac_line.perform_ransac_iteration(
        starting_points, max_distance=args.max_distance, max_trials=args.max_trials
    )
    if len(inliers) < args.min_inliers_lines:
        print(f"Too few inliers: {len(inliers)}, threshold={args.min_inliers_lines}")
        return starting_points, line_results, False
    starting_points = outliers
    line_results.append(ransac_line.RansacLineInfo(inliers, model))
    return starting_points, line_results, True


def detect_circle(starting_points, circle_results, w, h, args, radius_limit=0.02):
    inliers, outliers, model = ransac_circle.perform_ransac_iteration(
        starting_points,
        max_distance=args.max_distance,
        max_radius=min(w, h),
        max_trials=args.max_trials,
    )
    radius = model.params[-1]
    relative_radius = radius / min(w, h)
    relative_thresh = args.min_inliers_circles * relative_radius

    if relative_radius < radius_limit:
        print(f"r small, (w,h)={(w, h)},  r={radius}, relative_r={relative_radius}")
        return starting_points, circle_results, True
    if (relative_radius < 1) and (len(inliers) < relative_thresh):
        print(f"(w,h)={(w, h)},  r={radius}, relative_r={relative_radius}")
        return starting_points, circle_results, True
    starting_points = outliers
    circle_results.append(ransac_circle.RansacCircleInfo(inliers, model))
    return starting_points, circle_results, True


def detect_lines_and_circles(starting_points, w, h, args):
    line_results, circle_results = [], []
    find_lines, find_circles = True, True
    if args.joint:
        for _ in range(args.iterations):
            if len(starting_points) <= 3:
                print("too few starting points")
                break
            elif find_lines and find_circles:
                starting_points, line_results, find_lines = detect_line(
                    starting_points, line_results, args
                )
                starting_points, circle_results, find_circles = detect_circle(
                    starting_points, circle_results, w, h, args
                )
            elif find_lines:
                starting_points, line_results, find_lines = detect_line(
                    starting_points, line_results, args
                )
            elif find_circles:
                starting_points, circle_results, find_circles = detect_circle(
                    starting_points, circle_results, w, h, args
                )
            else:
                break
    else:
        for _ in range(args.iterations):
            if len(starting_points) <= 2:
                print("too few starting points")
                break
            if find_lines:
                starting_points, line_results, find_lines = detect_line(
                    starting_points, line_results, args
                )
            else:
                break
        for _ in range(args.iterations):
            if len(starting_points) <= 3:
                print("too few starting points")
                break
            if find_circles:
                starting_points, circle_results, find_circles = detect_circle(
                    starting_points, circle_results, w, h, args
                )
            else:
                break
    print(f"Found {len(circle_results)} circles and {len(line_results)} lines")

    return line_results, circle_results


def get_extension(args):
    extension = f"_{args.max_trials}_{args.iterations}_{args.min_inliers_lines}_{args.min_inliers_circles}_{int(args.max_distance)}"
    if args.joint:
        extension += "_joint"
    # if args.keep_true_indices:
    #     extension += "_no_outliers"
    if args.resized:
        extension += "_resized"

    return extension


draw_svg = True
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.config_path:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
            print(config)
        args.max_trials = config["max_trials"]
        args.min_inliers_lines = config["min_inliers_lines"]
        args.min_inliers_circles = config["min_inliers_circles"]
        args.max_distance = config["max_distance"]
        args.iterations = config["iterations"]
        args.edge_folder = config["edge_folder"]
        args.output_folder = config["output_folder"]
        args.input_folder = config["input_folder"]
        args.evaluate = config["evaluate"]
        args.joint = config["joint"]
        args.resized = config["resized"]
        args.percentile = config["percentile"]
        args.discard_large_circles = config["discard_large_circles"]

    input_folder = INPUT_DATADIR / args.input_folder
    edge_folder = INPUT_DATADIR / args.edge_folder
    annotation_folder = INPUT_DATADIR / "svgs"

    extension = get_extension(args)
    output_folder = OUTPUT_DATADIR / (args.output_folder + extension)

    os.makedirs(output_folder, exist_ok=True)
    if args.save_plots:
        image_output_folder = output_folder / f"plots/percentile_{args.percentile}"
        os.makedirs(image_output_folder, exist_ok=True)

    filename = output_folder / "input_args.json"
    with open(filename, "w") as f:
        json.dump(vars(args), f)
    info_folder = output_folder / args.output_info_folder
    os.makedirs(info_folder)



    rescale_dict = defaultdict(list) if args.resized else None

    for image_path in input_folder.glob("*.png"):
        image_name = os.path.basename(image_path)
        diagram_name = image_name[:-4]
        print(f"Processing {diagram_name}")
        starting_points, original_img = get_images(image_path, edge_folder=edge_folder)
        h, w = original_img.shape[:2]
        line_results, circle_results = detect_lines_and_circles(
            starting_points, w, h, args
        )
        save_info(info_folder, image_path, line_results, circle_results)
        if args.resized:  # FIXME:
            rescale_dict[diagram_name] = h
        if args.save_plots:
            if draw_svg:
                drawing = svgwrite.Drawing(str(), size=(w, h))
            else:
                drawing = None
            line_image, drawing_line = ransac_line.plot_all_inliers(
                line_results,
                w,
                h,
                original_img=original_img.copy(),
                percent=args.percentile,
                drawing=drawing,
            )
            all_image, drawing_all = ransac_circle.plot_all_inliers(
                circle_results,
                w,
                h,
                original_img=line_image.copy(),
                discard_large_circles=args.discard_large_circles,
                circle_ratio=0.5,
                drawing=drawing_line.copy(),
            )
            output_image_path = os.path.join(image_output_folder, image_name)
            cv2.imwrite(output_image_path, all_image)
            drawing_all.saveas(output_image_path[:-4] + ".svg")
    if args.evaluate:
        line_thresholds = args.line_thresholds
        circle_thresholds = args.circle_thresholds
        for line_threshold, circle_threshold in zip(line_thresholds, circle_thresholds):
            precision, recall, fscore, precision_c, recall_c, fscore_c = main_eval(
                info_folder,
                annotation_folder,
                line_threshold=line_threshold,
                circle_threshold=circle_threshold,
                rescale_dict=rescale_dict,
            )
            print(
                f"line precision {precision}, recall {recall}, fscore {fscore}, threshold {line_threshold}"
            )
            print(
                f"circle precision {precision_c}, recall {recall_c}, fscore {fscore_c}, threshold {circle_threshold}"
            )
