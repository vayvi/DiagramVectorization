import os
import torch
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import cv2
import json
from glob import glob
import argparse

# TODO: add arguments for input data (mainly -corr real data)


def get_bbox_from_center_radii(center, radii):
    center = np.array(center)
    radii = np.array(radii)
    xs = np.column_stack((center[:, 0] - radii, center[:, 0] + radii))
    ys = np.column_stack((center[:, 1] - radii, center[:, 1] + radii))
    bbox = np.dstack((xs, ys))

    return bbox


def get_positions(lines, im_shape, heatmap_scale):
    fy, fx = heatmap_scale[1] / im_shape[0], heatmap_scale[0] / im_shape[1]
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lpos, lnid, junc = [], [], []
    jids = {}

    def jid(jun):
        jun = tuple(jun[:2])  # (x,y)
        if jun in jids:  # if it already has an id, return it
            return jids[jun]
        jids[jun] = len(junc)  # otherwise, create an id for it
        junc.append(np.array(jun))  # add it to the list of junctions to update ids.
        return len(junc) - 1

    for v0, v1 in lines:
        lnid.append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])
    lpos = np.array(lpos, dtype=np.float32)
    Lpos = np.array(lnid, dtype=np.int16)
    return lpos, Lpos, junc


def main(data_path):
    batch = "valid"
    output_dir = data_path / f"{batch}_labels"
    os.makedirs(output_dir)
    anno_file = os.path.join(data_path, f"{batch}.json")

    with open(anno_file, "r") as f:
        dataset = json.load(f)

    for data in dataset:
        circles = get_bbox_from_center_radii(
            data["circle_centers"], data["circle_radii"]
        )
        im_path = str((data_path / "images") / data["filename"])
        image = cv2.imread(im_path)
        prefix = data["filename"].split(".")[0]
        lines = np.array(data["lines"]).reshape(-1, 2, 2)
        im_shape = image.shape

        im_rescale = (512, 512)
        heatmap_scale = (128, 128)

        lpos_c, Lpos_c, junc_c = get_positions(circles.copy(), im_shape, heatmap_scale)
        lpos, Lpos, junc = get_positions(lines.copy(), im_shape, heatmap_scale)
        image = cv2.resize(image, im_rescale)

        np.savez_compressed(
            output_dir / f"{prefix}.npz",
            aspect_ratio=image.shape[1] / image.shape[0],
            lines=lpos,
            circles=lpos_c,
        )
        cv2.imwrite(str(output_dir / f"{prefix}.png"), image)
        print(f"Saved {prefix}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../synthetic_dataset")

    args = parser.parse_args()
    args.data_path = Path(args.data_path)
    main(args.data_path)
