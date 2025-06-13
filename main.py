##
## EPITECH PROJECT, 2025
## Main
## File description:
## Main
##

import tensorflow as tf
from UNet import UNetDetector
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from data import resize_image

def main():
    img_dir = "./car_pictures/320_180/"
    model_path = "./unet_simple.weights.h5"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(f"Gpus are: {gpus}")

    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
        print(f"Directory added but no image yet")
        exit()

    img_files = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not img_files:
        print("No image")
        exit()

    detector = UNetDetector()

    if os.path.exists(model_path):
        detector.load(model_path)
        print("Model loaded")
    else:
        print("training in progress...")
        detector.train(img_files, epochs=15, batch_size=2)
        print("Zebi c'est la vie")
        detector.save(model_path)
        print("model saved")

    for i, path in enumerate(img_files[:3]):
        img = mpimg.imread(path)
        result = detector.predict(img)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(result, cmap="gray")
        plt.title("U-Net")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(img)
        plt.imshow(
            resize_image(result, (img.shape[0], img.shape[1])), alpha=0.4, cmap="hot"
        )
        plt.title("Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
