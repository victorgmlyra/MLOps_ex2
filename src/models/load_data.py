import numpy as np


def load_data(data_file):
    data_dict = np.load(data_file)
    data = list(zip(data_dict["images"], data_dict["labels"]))
    return data
