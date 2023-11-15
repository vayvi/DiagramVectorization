import os
from PIL import Image
import cv2
import numpy as np
import pickle


def save_info(info_folder, image_file, line_results, circle_results):
    image_name = os.path.basename(image_file)[:-4]
    output_folder = os.path.join(info_folder, image_name)
    os.makedirs(output_folder, exist_ok=True)
    for k, line in enumerate(line_results):
        filename = os.path.join(output_folder, image_name + f"_line{k}.pkl")
        with open(filename, "wb") as outp:
            pickle.dump(line, outp)
    for k, circle in enumerate(circle_results):
        filename = os.path.join(output_folder, image_name + f"_circle{k}.pkl")
        with open(filename, "wb") as outp:
            pickle.dump(circle, outp)


def get_edge_points_from_edge_img(edge_img_path):

    img = cv2.imread(str(edge_img_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    indices = np.where(img > 1e-3)
    height = img.shape[0]
    width = img.shape[1]

    cartesian_y = height - 1 - indices[0]  # inverting y indices to get coordinates
    np_data_points = np.column_stack((indices[1], cartesian_y))  # indices[1] are x-axis
    return np_data_points, width, height


def get_edge_points(img_path: str):
    """
    Returns the coordinates of edges in cartesian basis
    """

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    edges_canny = cv2.Canny(img, 100, 200)
    indices = np.where(edges_canny != 0)
    height = edges_canny.shape[0]
    width = edges_canny.shape[1]

    cartesian_y = height - 1 - indices[0]  # inverting y indices to get coordinates
    np_data_points = np.column_stack((indices[1], cartesian_y))
    return np_data_points, width, height


def get_angles(cos_values, sin_values):
    mask = sin_values >= 0
    return np.where(mask, np.arccos(cos_values), -np.arccos(cos_values) + 2 * np.pi)


def get_images(image_path, edge_folder=None):  # TODO: clean this
    im_path = str(image_path)
    image_name = os.path.basename(im_path)
    original_img = cv2.imread(im_path)
    edge_image_path = edge_folder / image_name
    starting_points, w, h = get_edge_points_from_edge_img(edge_image_path)
    return starting_points, original_img
