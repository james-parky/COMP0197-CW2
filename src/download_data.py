"""
Module for downloading and splitting the Oxford Pet Dataset.
"""
import os
import tarfile
import shutil
import random

import requests
from PIL import Image
import numpy as np
import h5py

SEGMENTATION_DIR = "annotations/trimaps"
IMAGES_DIR = "images"
DATA_PATH = "./data"
H5_PATHS = [
    "images_test.h5",
    "images_train.h5",
    "images_val.h5",
    "labels_test.h5",
    "labels_train.h5",
    "labels_val.h5",
    "split.config.txt",
]
FILENAMES = ["images.tar.gz", "annotations.tar.gz"]
URL_BASE = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"


def already_downloaded() -> bool:
    """
    Check whether the correct image and label files already exist.

    Returns:
        (bool): Whether the files already exist.
    """
    return all(os.path.exists(f"{DATA_PATH}/{path}") for path in H5_PATHS)


def create_dir() -> None:
    """
    Create the .data directory to hold the image and label files if it doesn't already
    exist. Empty it if it exists already.
    """
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH)


def download_data() -> None:
    """
    Download the data from the url base linked at the top of this module.
    """
    print("Downloading and extracting data...")
    for temp_file in FILENAMES:
        url = URL_BASE + temp_file
        print(url + " ...")
        r = requests.get(url, allow_redirects=True, timeout=10)
        with open(temp_file, "wb") as temp:
            temp.write(r.content)
        with tarfile.open(temp_file) as tar_obj:
            tar_obj.extractall()
            tar_obj.close()
        os.remove(temp_file)


def split_data(ratio_train: float, ratio_val: float, ratio_test: float) -> None:
    """
    Convert the already downloaded temp files into .h5 dataset files, after splitting
    the dataset into the provided ratios.

    Args:
        ratio_train (float): The portion of the dataset reserved for training.
        ratio_val (float): The portion of the dataset reserved for validation.
        ratio_test (float): The portion of the dataset reserved for testing.
    """
    # ----- options -----
    im_size = (64, 64)
    # -------------------
    img_h5s, seg_h5s = [], []
    for s in ["train", "val", "test"]:
        img_h5s.append(
            h5py.File(os.path.join(DATA_PATH, f"images_{s:s}.h5"), "w")
        )
        seg_h5s.append(
            h5py.File(os.path.join(DATA_PATH, f"labels_{s:s}.h5"), "w")
        )

    img_filenames = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
    num_data = len(img_filenames)
    num_val = int(num_data * ratio_val)
    num_test = int(num_data * ratio_test)
    num_train = num_data - num_val - num_test

    print(
        f"Extracting data into {num_train}-{num_val}-{num_test} for train-val-test..."
    )

    random.seed(90)
    random.shuffle(img_filenames)

    # write all images/labels to h5 file
    for idx, im_file in enumerate(img_filenames):
        if idx < num_train:  # train
            ids = 0
        elif idx < (num_train + num_val):  # val
            ids = 1
        else:  # test
            ids = 2

        with Image.open(os.path.join(IMAGES_DIR, im_file)) as img:
            img = np.array(
                img.convert("RGB").resize(im_size).getdata(), dtype="uint8"
            ).reshape((im_size[0], im_size[1], 3))
            img_h5s[ids].create_dataset(f"{idx:06d}", data=img)
        with Image.open(
            os.path.join(SEGMENTATION_DIR, im_file.split(".")[0] + ".png")
        ) as seg:
            seg = np.array(
                seg.resize(im_size).getdata(), dtype="uint8"
            ).reshape(im_size[0], im_size[1])
            seg_h5s[ids].create_dataset(f"{idx:06d}", data=seg)

    for ids, _ in enumerate(img_h5s):
        _.flush()
        _.close()
        seg_h5s[ids].flush()
        seg_h5s[ids].close()

    shutil.rmtree(IMAGES_DIR)
    shutil.rmtree(
        SEGMENTATION_DIR.split("/", maxsplit=1)[0]
    )  # remove entire annatations folder

    print(f"Data saved in {os.path.abspath(DATA_PATH)}")
    with open(
        f"{DATA_PATH}/split.config.txt", "w+", encoding="utf-8"
    ) as config_file_w:
        config_file_w.write(f"{ratio_train}\n")
        config_file_w.write(f"{ratio_val}\n")
        config_file_w.write(f"{ratio_test}")


if __name__ == "__main__":
    if already_downloaded():
        with open(
            f"{DATA_PATH}/split.config.txt", "r", encoding="utf-8"
        ) as config_file_r:
            splits = [next(config_file_r) for _ in range(3)]
            print(
                f"Dataset already downloaded with the following split:\n"
                f"\tTraining: {splits[0]}\n"
                f"\tValidation: {splits[1]}\n"
                f"\tTesting: {splits[2]}"
            )
    else:
        create_dir()
        download_data()
        split_data(0.7, 0.1, 0.2)
