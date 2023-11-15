import numpy as np
from svg.path import parse_path
from svg.path.path import Line, Move, Arc
from xml.dom import minidom
import glob
import json
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/diagrams")


class BadPath(Exception):
    pass


def read_path(path_strings):
    lines = []
    # print the line draw commands
    for path_string in path_strings:
        path = parse_path(path_string)

        for e in path:
            if isinstance(e, Line):
                x0, y0 = e.start.real, e.start.imag
                x1, y1 = e.end.real, e.end.imag
                line = np.array([x0, y0, x1, y1])
                if np.linalg.norm(line) > 1e-5:
                    lines.append(line)
            elif not isinstance(e, Move):
                raise BadPath(f"Path contains a curve/arc or an unknown object: {e}")
    return lines


def get_radius(circle):
    try:
        r = float(circle.getAttribute("r"))
    except ValueError as e:
        print(e)
        try:
            rx = float(circle.getAttribute("rx"))
            ry = float(circle.getAttribute("ry"))
            if np.abs(1 - ry / rx) < 1e-2:
                r = (rx + ry) / 2
            else:
                raise BadPath(f"shape is coded as circle with rx,ry={rx} , {ry}")
        except Exception:
            raise BadPath(f"Invalid circle")
    return r


def get_gt_from_svg(annotation_path, ellipse_to_circle_ratio_threshold=0.05):
    doc = minidom.parse(str(annotation_path))
    svg_params = doc.getElementsByTagName("svg")[0]
    width, height = svg_params.getAttribute("width"), svg_params.getAttribute("height")
    image_link = doc.getElementsByTagName("image")[0].getAttribute("xlink:href")

    lines, circles, ellipses, circle_r, circle_centers = [], {}, {}, [], []
    doc_circles, doc_ellipses = doc.getElementsByTagName(
        "circle"
    ), doc.getElementsByTagName("ellipse")
    if len(doc_circles) > 0:
        circle_r = np.array([get_radius(circle) for circle in doc_circles])

        circle_centers = np.array(
            [
                [float(circle.getAttribute("cx")), float(circle.getAttribute("cy"))]
                for circle in doc_circles
            ]
        )
    if len(doc_ellipses) > 0:
        print("inside ellipse")  # TODO: remove prints after testing
        print(annotation_path)
        ellipse_centers = np.array(
            [
                [float(ellipse.getAttribute("cx")), float(ellipse.getAttribute("cy"))]
                for ellipse in doc_ellipses
            ]
        )
        ellipse_r = np.array(
            [
                [float(ellipse.getAttribute("rx")), float(ellipse.getAttribute("ry"))]
                for ellipse in doc_ellipses
            ]
        )
        mask = (
            np.abs((ellipse_r[:, 0] / (ellipse_r[:, 1] + 1e-8)) - 1)
            < ellipse_to_circle_ratio_threshold
        )
        circle_centers = np.vstack([circle_centers, ellipse_centers[mask]])
        circle_r = np.concatenate([circle_r, np.mean(ellipse_r[mask], axis=1)])
        ellipse_centers, ellipse_r = ellipse_centers[~mask], ellipse_r[~mask]
        if len(ellipse_centers) > 0:
            ellipses = {"ellipse_centers": ellipse_centers, "ellipse_radii": ellipse_r}
            print(f"svg {annotation_path} has ellipses.")

    if len(circle_centers) > 0:
        circles = {"centers": circle_centers, "radii": circle_r}

    path_strings = [path.getAttribute("d") for path in doc.getElementsByTagName("path")]

    doc.unlink()

    lines = read_path(path_strings)

    return {
        "line_coords": np.array(lines),
        "circle_pos": circles["centers"],
        "circle_radius": circles["radii"],
        "ellipses": ellipses,
        "width": float(width),
        "height": float(height),
    }


def get_annotation(table):
    centers, circle_radii, lines = [], [], []
    for circle_pos, circle_radius in zip(table["circle_pos"], table["circle_radius"]):
        center = [circle_pos[0], circle_pos[1]]
        centers.append(center)
        circle_radii.append(circle_radius)
    for line_coords in table["line_coords"]:
        lines.append(
            [
                float(line_coords[0]),
                float(line_coords[1]),
                float(line_coords[2]),
                float(line_coords[3]),
            ]
        )

    return {
        "circle_centers": centers,
        "circle_radii": circle_radii,
        "lines": lines,
        "width": table["width"],
        "height": table["height"],
    }


if __name__ == "__main__":
    args = parser.parse_args()
    parent_folder_path = Path(args.data_path)

    annotations = []
    print(parent_folder_path / "svgs")
    for svg_path in glob.glob(str(parent_folder_path / "svgs") + "/*-corr.svg"):
        print("processing", svg_path)
        name = os.path.basename(svg_path).split("-corr.svg")[0]
        try:
            table = get_gt_from_svg(svg_path)
        except BadPath as e:
            print(e)
            print(f"Skipping {svg_path}")
            continue
        annotation = get_annotation(table)
        annotation["filename"] = f"{name}.png"
        annotations.append(annotation)

    with open(parent_folder_path / "valid.json", "w") as json_file:
        json.dump(annotations, json_file)
