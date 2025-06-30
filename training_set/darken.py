from PIL import ImageChops, Image
import os

input_folder = "vanilla"
output_folder = "lighter"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        darken = Image.new("RGB", img.size, (120, 120, 120))
        darkened_img = ImageChops.add(img, darken)
        darkened_img.save(os.path.join(output_folder, filename))