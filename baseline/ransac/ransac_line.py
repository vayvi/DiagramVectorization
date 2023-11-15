# Builds on https://medium.com/mlearning-ai/recursive-ransac-approach-to-find-all-straight-lines-in-an-image-b5c510a0224a and https://github.com/scikit-image/scikit-image/blob/441fe68b95a86d4ae2a351311a0c39a4232b6521/skimage/measure/fit.py#L29

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.measure import LineModelND
from skimage.measure import ransac as ransac_skimage
import argparse
from ransac.utils import get_edge_points, get_edge_points_from_edge_img
import drawsvg as draw

MIN_SAMPLES_LINE = 2  # TODO: check if it works with 2 and not 3


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_folder",
        type=str,
        default="sample/",
        help="folder where images are located.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="out/ransac/lines/",
        help="folder where images will be saved",
    )
    parser.add_argument(
        "--input_image", type=str, default="cropped_diagram.png", help="image name"
    )
    parser.add_argument(
        "--ignore_existing_predictions",
        action="store_true",
        help="ignores images where ransac has already been computed.",
    )
    parser.add_argument(
        "--precomputed_edges",
        action="store_true",
        help="Use precomputed edge images notably for images where text has already been removed.",
    )
    parser.add_argument(
        "--edge_folder",
        type=str,
        default="out/removed_text/",
        help="folder where edge images exist",
    )
    parser.add_argument(
        "--test_on_one_example",
        action="store_true",
        help="performs ransac line detection over one image name given in input_image instead of entire image",
    )
    return parser


class RansacLineInfo(object):
    """Helper class to manage the information about the RANSAC line.

    Attributes:
        inliers (numpy.ndarray): The inlier points that were detected by the RANSAC algorithm.
        model (LineModelND): The linear model that was the result of the RANSAC algorithm.
    """

    def __init__(self, inliers: np.ndarray, model: LineModelND):
        self.inliers = inliers
        self.model = model

    @property
    def unitvector(self):
        """The unit vector of the model. This is an array of 2 elements (x,y)."""
        return self.model.params[1]


def perform_ransac_iteration(
    data_points, max_distance: int, model_class=LineModelND, max_trials=500
):


    model_robust, inlier_indices = ransac_skimage(
        data_points,
        model_class,
        min_samples=MIN_SAMPLES_LINE,
        residual_threshold=max_distance,
        max_trials=max_trials,
    )
    inliers = data_points[inlier_indices]
    outliers = data_points[~inlier_indices]
    return inliers, outliers, model_robust


def compute_points_on_segment(
    model: LineModelND, x_min: int, x_max: int, y_min: int, y_max: int
):
    unit_vector = model.params[1]
    slope = abs(unit_vector[1] / unit_vector[0])
    if slope > 1:
        y_values = np.arange(y_min, y_max, 1)
        x_values = model.predict_x(y_values)
    else:
        x_values = np.arange(x_min, x_max, 1)
        y_values = model.predict_y(x_values)

    np_data_points = np.column_stack((x_values, y_values))
    return np_data_points


def plot_all_inliers(
    ransac_lines,
    width: float,
    height: float,
    original_img: np.ndarray,
    percent: int = 5,
    drawing=None,
) -> np.ndarray:
    new_image = original_img
    lines_only_image = np.zeros(new_image.shape[:2])
    for ransac_line_info in ransac_lines:
        color = (0, 0, 255)
        inliers = ransac_line_info.inliers
        indices = np.argsort(inliers[:, 0], axis=0)
        inliers = inliers[indices]
        lower, upper = np.percentile(inliers, [percent, 100 - percent], axis=0)

        plottable_points = compute_points_on_segment(
            ransac_line_info.model,
            x_min=lower[0],
            x_max=upper[0],
            y_min=lower[1],
            y_max=upper[1],
        )

        for point in plottable_points:
            # ### to remove
            # x = int(round(point[0]))
            # if (x >= width) or (x < 0):
            #     continue
            # y = int(round(point[1]))
            # if (y >= height) or (y < 0):
            #     continue
            # ###
            new_y = height - y - 1
            lines_only_image[new_y][x] = color[0]
            new_image[new_y][x][0] = color[0]
            new_image[new_y][x][1] = color[1]
            new_image[new_y][x][2] = color[2]
        p1, p2 = (
            plottable_points[0],
            plottable_points[-1],
        )
        cartesian = lambda point: [point[0], (height - point[1] - 1)]
        p1, p2 = cartesian(p1), cartesian(p2)
        if drawing is not None:
            drawing.add(drawing.line(start=p1, end=p2, stroke="red", stroke_width=3))
    return new_image, drawing


def detect_lines(
    image_path: str,
    iterations: int,
    max_distance: int,
    min_inliers_allowed: int,
    output_folder: str,
    ignore_existing_predictions=False,
    precomputed_edges=False,
    edge_folder=None,
) -> None:
    image_name = os.path.basename(image_path)
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, image_name)
    if os.path.exists(output_image_path) and ignore_existing_predictions:
        print(f"found existing prediciton for {image_name}")
        return None

    original_img = cv2.imread(image_path)
    results = []
    if precomputed_edges:
        edge_image_path = os.path.join(edge_folder, image_name)
        starting_points, width, height = get_edge_points_from_edge_img(edge_image_path)
        original_img = cv.resize(original_img, (width, height))
    else:
        starting_points, width, height = get_edge_points(image_path)

    for _ in range(iterations):
        if len(starting_points) <= MIN_SAMPLES_LINE:
            break

        inliers, outliers, model = perform_ransac_iteration(
            starting_points, max_distance=max_distance
        )
        if len(inliers) < min_inliers_allowed:
            print(f"Too few inliers: {len(inliers)}, threshold={min_inliers_allowed}")
            break

        starting_points = outliers
        results.append(RansacLineInfo(inliers, model))
        print("Found %d RANSAC lines" % (len(results)))

    superimposed_image, detection_only_image = plot_all_inliers(
        results, width, height, original_img=original_img
    )

    io.imsave(output_image_path, superimposed_image)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.test_on_one_example:
        detect_lines(
            os.path.join(args.input_folder, args.input_image),
            iterations=15,
            max_distance=1,
            min_inliers_allowed=200,
        )
    else:
        for image_path in os.listdir(args.input_folder):
            detect_lines(
                os.path.join(args.input_folder, image_path),
                iterations=15,
                max_distance=1,
                min_inliers_allowed=200,
                output_folder=args.output_folder,
                ignore_existing_predictions=args.ignore_existing_predictions,
                precomputed_edges=args.precomputed_edges,
                edge_folder=args.edge_folder,
            )
