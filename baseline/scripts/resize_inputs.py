import os
import glob

from pathlib import Path
from tqdm import tqdm
from ransac import INPUT_DATADIR, max_dimension
from PIL import Image


def resize_image(image_path, max_dimension):
    image = Image.open(image_path)
    width, height = image.size

    if max(width, height) <= max_dimension:
        return image
    if width > height:
        new_width = max_dimension
        new_height = int((height / width) * max_dimension)
    else:
        new_width = int((width / height) * max_dimension)
        new_height = max_dimension

    resized_image = image.resize((new_width, new_height))

    return resized_image


if __name__ == "__main__":
    print(INPUT_DATADIR)
    output_dir = INPUT_DATADIR / "images_resized"
    os.makedirs(output_dir, exist_ok=True)

    for image_path in tqdm(glob.glob(str(INPUT_DATADIR / "images/*.png"))):
        print("resizing", image_path)
        image_name = os.path.basename(image_path)
        image_name = image_name[:-4]
        output_img_path = str(output_dir / f"{image_name}.png")
        resized_image = resize_image(image_path, max_dimension)
        resized_image.save(output_img_path)
