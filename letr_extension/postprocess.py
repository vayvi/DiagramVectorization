import torch
import pytorch_lightning as pl
import yaml
from LETR.model.letr import LETR
from glob import glob
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as functional
from types import SimpleNamespace


def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


def load_model(cfg, exp_folder=None, ckpt_path=None):
    # print(exp_folder)
    if exp_folder is not None:
        for checkpoint in sorted(glob(str(exp_folder / "*.ckpt"))):
            print(checkpoint)
            checkpoint = torch.load(checkpoint, map_location="cpu")
            # break
    elif ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    else:
        raise ValueError("exp_folder or ckpt_path must be provided")

    model = LETR(
        backbone_cfg=cfg.letr.backbone,
        position_cfg=cfg.letr.position_encoding,
        transformer_stage1_cfg=cfg.letr.transformer_stage1,
        criterion_cfg=cfg.letr.criterion,
        letr_cfg=cfg.letr,
        transformer_stage2_cfg=cfg.letr.transformer_stage2,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


from helper.misc import nested_tensor_from_tensor_list


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image


class ToTensor(object):
    def __call__(self, img):
        return functional.to_tensor(img)


def resize(image, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = functional.resize(image, size)

    return rescaled_image


class Resize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = self.sizes
        return resize(img, size, self.max_size)


def process_image(raw_img):
    h, w = raw_img.shape[0], raw_img.shape[1]
    print(h, w)
    orig_size = torch.as_tensor([int(h), int(w)])

    # normalize image
    test_size = 1100
    normalize = Compose(
        [
            ToTensor(),
            Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273]),
            Resize([test_size]),
        ]
    )
    img = normalize(raw_img)
    inputs = nested_tensor_from_tensor_list([img])
    return inputs, orig_size


CLASSES = ["lines", "circles"]


def plot_results(
    pil_img, prob, boxes, thresh_line, thresh_circle, relative=False, plot_proba=False
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 30))
    ax1.imshow(pil_img, aspect="equal")
    plt.axis("off")

    ax2.imshow(pil_img, aspect="equal")
    plt.axis("off")

    ax3.imshow(pil_img, aspect="equal")
    # ax = plt.gca()
    for p, line in zip(prob, boxes):
        ymin, xmin, ymax, xmax = line.detach().numpy()
        cl = p.argmax()
        if cl == 0:
            if p.max() > thresh_line:
                ax2.plot(
                    [xmin, xmax],
                    [ymin, ymax],
                    linewidth=1,
                    color="green",
                    zorder=1,
                )

                text = f"l: {p[cl]:0.2f}"
                if plot_proba:
                    ax2.text(
                        xmin,
                        ymin,
                        text,
                        fontsize=15,
                        bbox=dict(facecolor="yellow", alpha=0.3),
                    )
        else:
            if relative:
                ymax += ymin
                xmax += xmin
            if p.max() > thresh_circle:
                r1 = (xmax - xmin) / 2
                r2 = (ymax - ymin) / 2
                if r1 * r2 > 0:
                    center = (xmin + r1, ymin + r2)
                    ax3.add_patch(
                        plt.Circle(center, r2, color="green", fill=False, linewidth=1)
                    )
                    if plot_proba:
                        ax3.text(
                            center[0],
                            center[1] + r1,
                            f"p={p[cl]:.2f}",
                            fontsize=15,
                            bbox=dict(facecolor="yellow", alpha=0.1),
                        )
                else:
                    if r1 > 0:
                        center = (xmin + r1, ymin - r1)
                        ax3.add_patch(
                            plt.Circle(
                                center, r1, color="green", fill=False, linewidth=1
                            )
                        )
                        if plot_proba:
                            ax3.text(
                                center[0],
                                center[1] + r1,
                                f"p={p[cl]:.2f}",
                                fontsize=15,
                                bbox=dict(facecolor="yellow", alpha=0.1),
                            )
                    else:
                        center = (xmin - r2, ymin + r2)

                        ax3.add_patch(
                            plt.Circle(
                                center, r2, color="blue", fill=False, linewidth=1
                            )
                        )
                        if plot_proba:
                            ax3.text(
                                center[0],
                                center[1] + r2,
                                f"p={p[cl]:.2f}",
                                fontsize=15,
                                bbox=dict(facecolor="yellow", alpha=0.1),
                            )

    plt.axis("off")
    plt.show()
    return fig
