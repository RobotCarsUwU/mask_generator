##
## EPITECH PROJECT, 2025
## Main
## File description:
## Main
##

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from UNet import UNetDetector
import os
import matplotlib.image as mpimg

from data import resize_image


def main():
    vanilla_dir = "training_set/vanilla"
    masked_dir  = "training_set/masked"
    test_dir = "tests_data"

    model_path = "./unet_simple.weights.h5"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(f"Gpus are: {gpus}")

    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    vanilla_files = sorted([
        os.path.join(vanilla_dir, f)
        for f in os.listdir(vanilla_dir)
        if f.lower().endswith((".png",".jpg",".jpeg"))
    ])
    masked_files = sorted([
        os.path.join(masked_dir, f)
        for f in os.listdir(masked_dir)
        if f.lower().endswith((".png",".jpg",".jpeg"))
    ])
    test_data = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".png",".jpg",".jpeg"))
    ])

    detector = UNetDetector()

    if os.path.exists(model_path):
        detector.load(model_path)
        print("Model loaded")
    else:
        print("training in progress...")
        detector.train(vanilla_files, masked_files, epochs=20, batch_size=16)
        print("Zebi c'est la vie")
        detector.save(model_path)
        print("model saved")

    for i, path in enumerate(test_data):
        img = mpimg.imread(path)
        print("Begin prediction...")
        result = detector.predict(img)
        mpimg.imsave(path + ".out.png", result)
        print("Prediction done")

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


