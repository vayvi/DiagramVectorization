import numpy as np
from svg.path import parse_path
from svg.path.path import Line, Move
from xml.dom import minidom


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
                line = np.array([[x0, y0], [x1, y1]])
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


def get_gt_from_svg(annotation_path, ellipse_to_circle_ratio_threshold=1e-2):
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
        print("inside ellipse")
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
        print(ellipse_r)
        print((ellipse_r[:, 0] / (ellipse_r[:, 1] + 1e-8)))
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
        "lines": np.array(lines),
        "circles": circles,
        "ellipses": ellipses,
        "image_link": image_link,
        "width": float(width),
        "height": float(height),
    }
