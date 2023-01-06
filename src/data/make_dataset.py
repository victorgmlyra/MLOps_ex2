# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import glob
import numpy as np
import torch
from torchvision import transforms


def normalize(data):
    data_t = torch.from_numpy(data)

    # shape of images = [b,c,w,h]
    mean, std = data_t.mean([0, 1, 2]), data_t.std([0, 1, 2])
    norm_data = transforms.Normalize(mean=mean, std=std)(data_t)

    norm_data = norm_data.numpy()
    return norm_data


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Train data
    train_data, train_labels = [], []
    train_files = sorted(glob.glob(input_filepath + "/train_*.npz"))
    for f in train_files:
        train_dict = np.load(f)
        train_data += list(train_dict["images"])
        train_labels += list(train_dict["labels"])
    train_data = normalize(np.array(train_data))
    train_labels = np.array(train_labels)

    # Test data
    test_dict = np.load("data/raw/test.npz")
    test_data = np.array(list(test_dict["images"]))
    test_norm_data = normalize(test_data.copy())
    test_labels = np.array(list(test_dict["labels"]))

    np.savez(output_filepath + "/train.npz", images=train_data, labels=train_labels)
    np.savez(
        output_filepath + "/test.npz",
        images=test_norm_data,
        labels=test_labels,
        ori_imgs=test_data,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
