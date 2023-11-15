import pickle
import numpy as np
from ransac.ransac_line import compute_points_on_segment


def parse_pkl(prediction_dir, width, height, percentile=1, radius_limit_postprocess=0.6):
    cartesian = lambda point: np.array([point[0], (height - point[1] - 1)])

    pred_lines, centers, radii = [], [], []
    for line_file in prediction_dir.glob("*_line*.pkl"):
        with open(line_file, "rb") as f:
            ransac_line_info = pickle.load(f)

        inliers = ransac_line_info.inliers
        indices = np.argsort(inliers[:, 0], axis=0)
        inliers = inliers[indices]
        lower, upper = np.percentile(inliers, [percentile, 100 - percentile], axis=0)
        plottable_points = compute_points_on_segment(
            ransac_line_info.model,
            x_min=lower[0],
            x_max=upper[0],
            y_min=lower[1],
            y_max=upper[1],
        )
        pred_lines.append(
            [cartesian(plottable_points[0]), cartesian(plottable_points[-1])]
        )

    for circle_file in prediction_dir.glob("*_circle*.pkl"):
        with open(circle_file, "rb") as f:
            ransac_circle_info = pickle.load(f)
        xc, yc, r = ransac_circle_info.model.params
        if r > radius_limit_postprocess * min(height, width):
            print(
                f"discarding circle with c=({np.round(xc,2)},{np.round(yc,2)}) and r={np.round(r, 2)} where image size= ({np.round(height,2)},{np.round(width,2)}))"
            )
            continue
        centers.append(cartesian([xc, yc]))
        radii.append(r)
    pred_circles = {"centers": np.array(centers), "radii": np.array(radii)}

    return np.array(pred_lines), pred_circles
