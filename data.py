##
## EPITECH PROJECT, 2025
## Data
## File description:
## Data
##

from PIL import Image, ImageDraw
import matplotlib.image as mpimg
import numpy as np

def create_line_mask(image):
    height, width = image.shape[:2]
    vertices = [(0, height - 1), (width // 2, height // 3), (width - 1, height - 1)]
    
    mask_img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask_img).polygon(vertices, fill=255)
    mask = np.array(mask_img)
    
    if len(image.shape) == 3:
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        gray = image
    
    masked_gray = np.where(mask == 255, gray, 0)
    line_mask = np.where(masked_gray > 200, 255, 0).astype(np.uint8)
    
    return line_mask

def resize_image(image, new_shape):
    img_pil = Image.fromarray(image)
    img_resized = img_pil.resize((new_shape[1], new_shape[0]), Image.BILINEAR)
    return np.array(img_resized)

def data_generator(image_paths, input_size=(180,320), batch_size=32):
    batch_X, batch_y = [], []
    for path in image_paths:
        img = mpimg.imread(path)
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        img = resize_image(img, (input_size[0], input_size[1]))
        mask = create_line_mask(img)

        batch_X.append(img / 255.0)
        batch_y.append(np.expand_dims(mask / 255.0, axis=-1))

        if len(batch_X) == batch_size:
            yield np.array(batch_X), np.array(batch_y)
            batch_X, batch_y = [], []

    if batch_X:
        yield np.array(batch_X), np.array(batch_y)
