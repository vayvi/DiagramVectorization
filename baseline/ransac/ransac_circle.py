# Builds on https://medium.com/mlearning-ai/recursive-ransac-approach-to-find-all-straight-lines-in-an-image-b5c510a0224a and https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/measure/fit.py#L220-L377

import os
import drawsvg as draw

import cv2 as cv
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.measure import CircleModel
from skimage.measure import ransac as ransac_skimage
import argparse
import warnings
from ransac.utils import (
    get_edge_points_from_edge_img,
    get_edge_points,
)

MIN_SAMPLES_CIRCLE = 3
# TODO: put this in a get_parser function


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
        default="out/ransac/circles/",
        help="folder where images will be saved",
    )
    parser.add_argument(
        "--input_image", type=str, default="cropped_diagram.png", help="image name"
    )
    parser.add_argument(
        "--min_inliers", type=int, default=200, help="Minimal number of inliners"
    )
    parser.add_argument(
        "--ignore_existing_predictions",
        action="store_true",
        help="ignores images where ransac has already been computed.",
    )

    parser.add_argument(
        "--test_on_one_example",
        action="store_true",
        help="performs ransac line detection over one image name given in input_image instead of entire image",
    )
    parser.add_argument(
        "--discard_large_circles",
        action="store_true",
        help="discard circles with too large radius (at least twice as large as maximal image size)",
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
    return parser


class RansacCircleInfo(object):
    """Helper class to manage the information about the RANSAC line.

    Attributes:
        inliers (numpy.ndarray): The inlier points that were detected by the RANSAC algorithm.
        model (LineModelND): The linear model that was the result of the RANSAC algorithm.
    """

    def __init__(self, inliers: np.ndarray, model: CircleModel):
        self.inliers = inliers
        self.model = model

    @property
    def params(self):
        """The unit vector of the model. This is an array of 2 elements (x,y)."""
        return self.model.params


def perform_ransac_iteration(
    data_points,
    max_distance: float,
    model_class=CircleModel,
    max_radius=1000,
    max_attempts=10,
    max_trials=500,
    keep_true_indices=True,
):
    """Performs one RANSAC iteration by fitting a line over input points.

    Args:
        data_points (np.array): An array with shape (N, 2) containing N points, with coordinates x=[0], y=[1].
        max_distance (int): The maximum distance from a point to the fitted line for it to be considered an inlier. The distance is r - np.sqrt((x - xc)**2 + (y - yc)**2)

    Returns:
        Tuple of:
            - inliers (np.array): A numpy array with shape (M, 2), containing the inliers of the just discovered RANSAC shape.
            - outliers (np.array): A numpy array with shape (N-M, 2), containing all data points with the inliers removed.
            - model_robust (_type_): A model representing the just discovered RANSAC line.
    """

    for i in range(max_attempts):
        model_robust, inlier_indices = ransac_skimage(
            data_points,
            model_class,
            min_samples=MIN_SAMPLES_CIRCLE,
            residual_threshold=max_distance,
            max_trials=max_trials,
        )
        if model_robust.params[-1] < max_radius:
            break
    else:
        print(f"found large circle with radius {model_robust.params[-1]}")
    if keep_true_indices:
        true_indices = np.abs(model_robust.residuals(data_points)) <= max_distance
        inliers = data_points[true_indices]
        outliers = data_points[~true_indices]
    else:
        inliers = data_points[inlier_indices]
        outliers = data_points[~inlier_indices]
    return inliers, outliers, model_robust


def compute_points_on_circle(model, t_min: float = 0, t_max: float = 2 * np.pi):
    angle_values = np.arange(t_min, t_max, 0.001)
    np_data_points = model.predict_xy(angle_values)
    return np_data_points


def plot_all_inliers(
    ransac_circles,
    width: float,
    height: float,
    original_img: np.ndarray,
    discard_large_circles=True,
    circle_ratio=0.6,
    drawing=None,
) -> np.ndarray:
    new_image = original_img.copy()
    circles_only_image = np.zeros(new_image.shape[:2])
    color = (0, 0, 255)

    for circle_index in range(0, len(ransac_circles)):
        ransac_circle_info = ransac_circles[circle_index]
        xc, yc, r = ransac_circle_info.model.params

        if discard_large_circles and (r > circle_ratio * min(width, height)):
            warnings.warn(
                f" discarding circle c=({np.round(xc,2)}, {np.round(yc,2)}) r={np.round(r, 2)} (h,w)=({np.round(height,2)},{np.round(width,2)})"
            )
            continue
        plottable_points = compute_points_on_circle(ransac_circle_info.model)

        for point in plottable_points:
            x = int(round(point[0]))
            if (x >= width) or (x < 0):
                continue
            y = int(round(point[1]))
            if (y >= height) or (y < 0):
                continue
            new_y = height - y - 1
            circles_only_image[new_y][x] = color[0]
            new_image[new_y][x][0] = color[0]
            new_image[new_y][x][1] = color[1]
            new_image[new_y][x][2] = color[2]
        if drawing is not None:
            drawing.add(
                drawing.circle(
                    center=[xc, height - yc - 1],
                    r=r,
                    fill="none",
                    stroke="red",
                    stroke_width=3,
                )
            )

    return new_image, drawing


def detect_circles(
    image_path: str,
    iterations: int,
    max_distance: float,  # RANSAC threshold distance from a line for a point to be classified as inlier
    min_inliers: int,
    output_folder: str,
    ignore_existing_predictions=False,
    discard_large_circles=True,
    precomputed_edges=False,
    edge_folder=None,
) -> None:
    print(f"Processing: {image_path}")
    image_name = os.path.basename(image_path)
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, image_name)
    if os.path.exists(output_image_path) and ignore_existing_predictions:
        print(f"found existing prediciton for {image_name}")
        return None

    original_img = cv.imread(image_path)
    print(original_img.shape)

    results = []
    if precomputed_edges:
        edge_image_path = os.path.join(edge_folder, image_name)
        starting_points, width, height = get_edge_points_from_edge_img(edge_image_path)
        print(width, height)
        original_img = cv.resize(original_img, (width, height))
        print(len(starting_points))
    else:
        starting_points, width, height = get_edge_points(image_path)
        print(f"Found {len(starting_points)} pixels in {image_path}")
    print(starting_points.shape, original_img.shape)
    max_radius = 2 * min(width, height)
    for _ in range(iterations):
        if len(starting_points) <= MIN_SAMPLES_CIRCLE:
            break

        inliers, outliers, model = perform_ransac_iteration(
            starting_points,
            max_distance=max_distance,
            max_radius=max_radius,
        )
        radius = model.params[-1]
        relative_radius = radius / min(
            width, height
        )  # TODO: think about documents where height and width are very different

        if relative_radius < 0.02:  # TODO: remove Hardcoding circle radius limit
            print(
                f"detected circle radius too small, (w,h)={(width, height)},  r={radius}, relative_r={relative_radius}"
            )
            break
        if (relative_radius < 1) and len(inliers) < min_inliers * 0.5 * relative_radius:
            print(
                f"Inliers: {len(inliers)}, original threshold={min_inliers}, adaptive threshold={min_inliers*0.5*relative_radius}"
            )
            print(f"(w,h)={(width, height)},  r={radius}, relative_r={relative_radius}")
            break

        starting_points = outliers
        results.append(RansacCircleInfo(inliers, model))
        print("Found %d RANSAC circles" % (len(results)))

    superimposed_image = plot_all_inliers(
        results,
        width,
        height,
        original_img=original_img,
        discard_large_circles=discard_large_circles,
    )

    io.imsave(output_image_path, superimposed_image)
    print(f"Results saved to file {output_image_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    output_folder = args.output_folder
    if args.test_on_one_example:
        detect_circles(
            f"sample/{args.input_image}",
            iterations=30,
            max_distance=1,
            min_inliers=200,
            output_folder=output_folder,
            discard_large_circles=args.discard_large_circles,
        )
    else:
        for image_path in os.listdir(args.input_folder):
            detect_circles(
                os.path.join(args.input_folder, image_path),
                iterations=15,
                max_distance=1.2,
                min_inliers=200,
                output_folder=output_folder,
                ignore_existing_predictions=args.ignore_existing_predictions,
                discard_large_circles=args.discard_large_circles,
                precomputed_edges=args.precomputed_edges,
                edge_folder=args.edge_folder,
            )
