from helper.image import paste_with_blured_borders, resize
from helper.path import coerce_to_path_and_check_exist
import numpy as np
from PIL import Image, ImageFilter
import cv2
import os
from numpy.random import uniform, choice
from random import randint, shuffle
from background import BackgroundElement
from diagram import DiagramElement
from helper.noise import get_random_noise_pattern
from helper.seed import use_seed
import json
from tqdm import tqdm
import PIL
from helper.color_utils import rgb_to_gray, img_8bit_to_float, gray_float_to_8bit
import matplotlib.pyplot as plt
from synthetic_module import DEFAULT_HEIGHT, DEFAULT_WIDTH

BLACK_AND_WHITE_FREQ = 0.1
BLUR_RADIUS_RANGE = (-0.1, 0.4)
BACKGROUND_BLURED_BORDER_WIDTH_RANGE = (1, 10)
LAYOUT_RANGE = {
    "nb_noise_patterns": (0, 5),
    "nb_words": (0, 10),
    "margin_h": (20, 100),
    "margin_v": (20, 100),
    "padding_h": (5, 80),
    "padding_v": (5, 80),
    "caption_padding_v": (0, 20),
    "context_margin_h": (0, 300),
    "context_margin_v": (0, 200),
}
NOISE_STD = 10
NOISE_PATTERN_RANGE = (0, 6)
SCALE_METHOD = [
    PIL.Image.NEAREST,
    PIL.Image.BOX,
    PIL.Image.BILINEAR,
    PIL.Image.HAMMING,
    PIL.Image.BICUBIC,
    PIL.Image.LANCZOS,
]


def xyxy_to_xyhw(lines=None, circles=None):
    new_lines_pairs = []
    if lines is not None:
        for line in lines:  # [ #lines, 2, 2 ]
            p1 = line[0]  # xy
            p2 = line[1]  # xy
            if p1[0] < p2[0]:
                new_lines_pairs.append([p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]])
            elif p1[0] > p2[0]:
                new_lines_pairs.append([p2[0], p2[1], p1[0] - p2[0], p1[1] - p2[1]])
            else:
                if p1[1] < p2[1]:
                    new_lines_pairs.append([p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]])
                else:
                    new_lines_pairs.append([p2[0], p2[1], p1[0] - p2[0], p1[1] - p2[1]])
    new_circles = []
    if circles is not None:
        centers, radii = circles

        for center, radius in zip(centers, radii):
            new_circles.append([center[0], center[1], radius, radius])

    return new_lines_pairs, new_circles


class SyntheticDiagram:
    def __init__(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        img_size=None,
    ) -> None:
        self.blur_radius = np.random.uniform(*BLUR_RADIUS_RANGE)
        # self.add_noise = choice([True, False], p=[0.8, 0.2])
        self.add_gaussian_noise = choice([True, False], p=[0.8, 0.2])
        self.add_resize_noise = choice([True, False], p=[0.8, 0.2])
        self.smooth = choice([True, False], p=[0.8, 0.2])

        self.black_and_white = choice(
            [True, False], p=[BLACK_AND_WHITE_FREQ, 1 - BLACK_AND_WHITE_FREQ]
        )
        self.background = BackgroundElement(width, height)
        self.noise_patterns = self._generate_random_noise_patterns()
        if img_size is not None:
            width, height = resize(
                Image.new("L", (width, height)), img_size, keep_aspect_ratio=True
            ).size  # TODO: understand this line
        self.width, self.height = width, height
        margin_h = randint(*LAYOUT_RANGE["margin_h"])
        margin_v = randint(*LAYOUT_RANGE["margin_v"])
        self.diagram_position = (margin_h, margin_v)
        self.diagram = DiagramElement(
            self.width, self.height, diagram_position=self.diagram_position
        )

    @use_seed()
    def _generate_random_noise_patterns(self):
        patterns, positions = [], []
        bg_width, bg_height = self.background.size
        for _ in range(randint(*NOISE_PATTERN_RANGE)):
            pattern, hue_color, value_ratio, position = get_random_noise_pattern(
                bg_width, bg_height
            )
            position = (position[0], position[1])
            patterns.append((pattern, hue_color, value_ratio))
            positions.append(position)
        return patterns, positions

    @property
    def size(self):
        return (self.width, self.height)

    def draw_noise_patterns(self, canvas):
        for (noise, hue_color, value_ratio), pos in zip(*self.noise_patterns):
            x, y = pos
            width, height = noise.size
            patch = np.array(canvas.crop([x, y, x + width, y + height]))
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            patch_hsv[:, :, 0] = hue_color
            patch_hsv[:, :, 2] = patch_hsv[:, :, 2] * value_ratio
            new_patch = Image.fromarray(cv2.cvtColor(patch_hsv, cv2.COLOR_HSV2RGB))
            canvas.paste(new_patch, pos, mask=noise)

    def to_image(self, path=None):
        canvas = Image.new(mode="RGB", size=self.size)
        background_img = self.background.to_image()
        paste_with_blured_borders(
            canvas,
            background_img,
            (0, 0),
            randint(*BACKGROUND_BLURED_BORDER_WIDTH_RANGE),
        )

        diagram_img = self.diagram.to_image()
        canvas.paste(diagram_img, (0, 0), mask=diagram_img)
        if self.add_resize_noise:
            scale_method = SCALE_METHOD[np.random.randint(0, 6)]
            factor = np.random.uniform(0.5, 1.5)
            small = canvas.resize(
                (
                    int(factor * self.width),
                    int(factor * self.height),
                ),  # TODO: remove hardcoded values
                scale_method,
            )
            scale_method = SCALE_METHOD[np.random.randint(0, 6)]
            same = small.resize((self.width, self.height), scale_method)
            repeated_noise = np.repeat(
                np.random.normal(loc=1, scale=0.008, size=(self.height, self.width))[
                    :, :, np.newaxis
                ],
                3,
                axis=2,
            )

            same = same * repeated_noise
            same = np.array(same).astype("uint8")
            canvas = Image.fromarray(same)

        elif self.add_gaussian_noise:

            def add_gaussian_noise(img):
                img_arr = np.array(img)
                noisy_img_arr = img_arr + np.random.normal(0, NOISE_STD, img_arr.shape)
                noisy_img_arr = np.clip(noisy_img_arr, 0, 255).astype(np.uint8)
                noisy_img = Image.fromarray(noisy_img_arr)
                return noisy_img

            canvas = add_gaussian_noise(canvas)
        if self.blur_radius > 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        if self.smooth:
            canvas = canvas.filter(ImageFilter.SMOOTH)
        if self.black_and_white:
            canvas = canvas.convert("L").convert("RGB")

        if path is not None:
            canvas.save(path)

        return canvas

    def save(self, name, output_dir):
        self._save_to_image(name, output_dir)
        self._save_to_svg(name, output_dir)

    def _save_to_svg(self, name, output_dir):
        output_dir = coerce_to_path_and_check_exist(output_dir)
        background_dir = output_dir / "backgrounds/"
        os.makedirs(background_dir, exist_ok=True)
        background_path = background_dir / f"{name}_background.png"

        self.background.to_image().save(background_path)
        svg_dir = output_dir / "svgs"
        os.makedirs(svg_dir, exist_ok=True)

        self.diagram.to_svg(
            svg_dir / f"{name}.svg", f"../backgrounds/{name}_background.png"
        )

    def _save_to_image(self, name, output_dir):
        output_dir = coerce_to_path_and_check_exist(output_dir)
        image_dir = output_dir / "images/"
        os.makedirs(image_dir, exist_ok=True)
        path = image_dir / f"{name}.png"
        img = self.to_image(path)

    def get_annotation(self, name):
        annotation = self.diagram.get_annotation()
        annotation["filename"] = f"{name}.png"
        annotation["height"] = self.height
        annotation["width"] = self.width
        return annotation

    def get_annotation_on_the_fly(self):
        data = self.diagram.get_annotation()
        annotation = []
        line_set, circle_set = None, None
        if len(data["lines"]) > 0:
            line_set = np.array(data["lines"], dtype=np.float64).reshape(-1, 2, 2)
        if len(data["circle_centers"]) > 0:
            circle_set = (data["circle_centers"], data["circle_radii"])

        line_set, circle_set = xyxy_to_xyhw(line_set, circle_set)
        anno_id = 0
        for line in line_set:
            info = {}
            info["id"] = anno_id
            anno_id += 1
            info["category_id"] = 0
            info["line"] = line
            annotation.append(info)

        for circle in circle_set:
            info = {}
            info["id"] = anno_id
            anno_id += 1
            info["category_id"] = 1
            info["circle"] = circle
            annotation.append(info)

        return annotation


class DiagramsDataset:
    def __init__(
        self,
        num_samples=1000,
        seed=None,
        train_val_split=0.2,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        offset=0,
    ) -> None:
        self.num_samples = num_samples
        self.seed = seed
        self.train_val_split = train_val_split
        self.width, self.height = width, height
        self.offset = offset

    def generate_dataset(self, img_size=None, output_dir="synthetic_dataset"):
        os.makedirs(output_dir, exist_ok=True)
        annotations = []
        output_dir = coerce_to_path_and_check_exist(output_dir)
        if self.offset:
            print("appending to existing dataset")
            assert (
                output_dir / "train.json"
            ).exists(), "training annotations not found"
            assert (
                output_dir / "valid.json"
            ).exists(), "validation annotations not found"
        for k in tqdm(range(self.offset, self.num_samples + self.offset, 1)):
            name = f"synthetic_diagram_{k}"
            synthetic_diagram = SyntheticDiagram(
                img_size=img_size,
                width=self.width,
                height=self.height,
            )
            synthetic_diagram.save(name, output_dir)
            annotations.append(synthetic_diagram.get_annotation(name))

        shuffle(annotations)
        val_size = int(self.train_val_split * (self.num_samples))
        valid_annotations = annotations[:val_size]
        train_annotations = annotations[val_size:]
        with open(output_dir / "train.json", "w") as json_file:
            json.dump(train_annotations, json_file)
        with open(output_dir / "valid.json", "w") as json_file:
            json.dump(valid_annotations, json_file)


if __name__ == "__main__":
    dataset = DiagramsDataset(num_samples=1000, seed=42)
    dataset.generate_dataset(output_dir="../../data/synthetic_raw")
