import cv2
from skimage import io
from pathlib import Path
from glob import glob
import os
import argparse
import numpy as np
from ransac import INPUT_DATADIR


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        default="images",
        help="Input folder of images.",
    )

    parser.add_argument(
        "--text_mask_folder",
        type=str,
        default="images_resized_testr_masks",
        help="Input folder of text masks.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output/edges",
        help="Output folder of edge images.",
    )
    parser.add_argument(
        "--hyst1",
        type=int,
        default=100,
        help="Lower hysteresis threshold for canny.",
    )
    parser.add_argument(
        "--hyst2",
        type=int,
        default=200,
        help="Upper hysteresis threshold for canny.",
    )
    parser.add_argument(
        "--blur_canny",
        type=int,
        default=3,
        help="Gaussian Blur kernel size.",
    )
    parser.add_argument(
        "--no_text_detection",
        action="store_true",
        help="If true, perform edge detection over original image without text removal step",
    )
    return parser


def canny(img, blur_kernel_size=3, hyst1=100, hyst2=200):
    blur_img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    edges_canny = cv2.Canny(blur_img, hyst1, hyst2, edges=True)
    return edges_canny


def save_contour_image(
    imagefilename,
    edge_path,
    blur_canny=3,
    hyst1=50,
    hyst2=200,
):
    img = cv2.imread(str(imagefilename), cv2.IMREAD_GRAYSCALE)
    assert (
        img is not None
    ), f"file {imagefilename} could not be read, check with os.path.exists()"
    edges_canny = canny(img, blur_canny, hyst1=hyst1, hyst2=hyst2)
    cv2.imwrite(edge_path, edges_canny)


def save_removed_text_contour_image(edge_path, mask_path, edge_path_wo_text):
    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE,
    )
    edge_image = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    assert np.shape(mask) == np.shape(
        edge_image
    ), f"no shape correspondance mask{np.shape(mask)} and contour{np.shape(edge_image)} "
    edge_image[mask > 0] = 0
    cv2.imwrite(edge_path_wo_text, edge_image)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    input_folder = INPUT_DATADIR / args.input_folder
    output_folder = INPUT_DATADIR / (args.output_folder + f"_{args.blur_canny}")

    os.makedirs(output_folder, exist_ok=True)

    if not args.no_text_detection:
        output_folder_wo_text = INPUT_DATADIR / (
            args.output_folder + f"_testr_{args.blur_canny}"
        )
        text_mask_folder = INPUT_DATADIR / f"{args.text_mask_folder}"
        os.makedirs(output_folder_wo_text, exist_ok=True)

    for image_path in (input_folder).glob("*.png"):
        image_name = os.path.basename(image_path)
        print(f"Processing: {image_name}")
        edge_path = output_folder / image_name
        mask_path = text_mask_folder / image_name

        save_contour_image(
            str(image_path),
            str(edge_path),
            args.blur_canny,
            args.hyst1,
            args.hyst2,
        )

        if not args.no_text_detection:
            edge_path_wo_text = output_folder_wo_text / image_name
            save_removed_text_contour_image(
                str(edge_path), str(mask_path), str(edge_path_wo_text)
            )
