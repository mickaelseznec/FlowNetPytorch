import bisect
import h5py
import numpy as np
import torch.utils.data as data
import linecache

from imageio import imread
from pathlib import Path


class HDFDataset(data.Dataset):
    def __init__(self, base_dir, transform=None, target_transform=None, co_transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

        self.h5file_list = self.compute_file_list(base_dir)
        self.cum_length = self.compute_cumsum_list(self.h5file_list)

    @staticmethod
    def compute_file_list(base_dir):
        return sorted(Path(base_dir).glob('*.hdf5'))

    @staticmethod
    def compute_cumsum_list(h5file_list):
        length_list = [0]

        for h5file in h5file_list:
            h5 = h5py.File(h5file, "r")

            key = next(iter(h5.keys())) #TODO: change this to handle multiple algorithms
            length_list.append(h5[key]["data"].shape[0])

        return np.cumsum(length_list)

    @staticmethod
    def get_hdf5_coords(index, cum_length):
        file_index =  bisect.bisect_left(cum_length, index + 1) - 1
        offset = index - cum_length[file_index]

        return file_index, offset

    def loader(self, target_h5_file, file_offset):
        h5 = h5py.File(target_h5_file, "r")

        key = next(iter(h5.keys()))
        flow = h5[key]["data"][file_offset, ...]

        image_path_list = h5.attrs["file_path"]
        image_path_list = image_path_list.replace("seznec", "seznecm")

        image_0, image_1 = linecache.getline(image_path_list, file_offset + 1).rstrip().split(",")

        image_0 = str(Path(image_path_list).parent / Path(image_0))
        image_1 = str(Path(image_path_list).parent / Path(image_1))

        return [imread(image_0).astype(np.float), imread(image_1).astype(np.float)], flow

    def __getitem__(self, index):
        target_h5_file, file_offset = self.get_hdf5_coords(index, self.cum_length)

        inputs, target = self.loader(self.h5file_list[target_h5_file], file_offset)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return self.cum_length[-1]

def hdf_dataset(*args, split=None, **kwargs):

    base_dataset = HDFDataset(*args, **kwargs)
    total_size = len(base_dataset)

    train_size = int(split * total_size)
    test_size = total_size - train_size
    return data.random_split(base_dataset, [train_size, test_size])

