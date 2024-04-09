"""
Small module to show the grayscale converter working through saving to .jpg
"""
import numpy as np
from PIL import Image
from download_data import DATA_PATH
from loader import H5ImageLoader

if __name__ == "__main__":
    images, labels = next(
        iter(
            H5ImageLoader(
                f"{DATA_PATH}/images_train.h5",
                10,
                f"{DATA_PATH}/labels_train.h5",
                True,
            )
        )
    )
    image_montage = Image.fromarray(
        np.concatenate([images[i] for i in range(len(images))], axis=1)
    )
    image_montage.save("train_images.jpg")
    label_montage = Image.fromarray(
        np.concatenate([labels[i] for i in range(len(labels))], axis=1)
    )
    label_montage.save("train_labels.jpg")
