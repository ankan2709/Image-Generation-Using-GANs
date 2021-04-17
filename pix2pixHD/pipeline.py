import os
from astropy.io import fits
from os.path import split, splitext
from random import randint
from glob import glob
import numpy as np
from natsort import natsorted
import torch
from scipy.ndimage import rotate
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, ToTensor, Normalize, Pad
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        dataset_dir = os.path.join('./datasets', 'AIA_to_HMI')
        self.input_format = opt.data_format_input
        self.target_format = opt.data_format_target

        if opt.is_train:
            self.label_path_list = natsorted(glob(os.path.join(dataset_dir, 'Train', 'Input', '*.' + self.input_format)))
            self.target_path_list = natsorted(glob(os.path.join(dataset_dir, 'Train', 'Target', '*.' + self.target_format)))

        else:
            self.label_path_list = natsorted(glob(os.path.join(dataset_dir, 'Test', 'Input', '*.' + self.input_format)))
            self.target_path_list = natsorted(glob(os.path.join(dataset_dir, 'Test', 'Target', '*.' + self.target_format)))

    def __getitem__(self, index):
        list_transforms = []
        list_transforms += []

        if self.opt.is_train:
            self.angle = randint(-self.opt.max_rotation_angle, self.opt.max_rotation_angle)

            self.offset_x = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            self.offset_y = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0

            if self.input_format in ["fits", "fts", "npy"]:
                if self.input_format in ["fits", "fts"]:
                    label_array = np.array(fits.open(self.label_path_list[index]))
                else:
                    label_array = np.load(self.label_path_list[index], allow_pickle=True)

                label_array = self.__rotate(label_array)
                # label_array = self.__pad(label_array, self.opt.padding_size)
                label_array = self.__random_crop(label_array)
                label_array = self.__convert_range(label_array,
                                                   min=-self.opt.dynamic_range_input,
                                                   max=2 * self.opt.dynamic_range_input)
                label_tensor = torch.tensor(label_array)
                label_tensor -= 0.5
                label_tensor = label_tensor / 0.5

                if len(label_tensor.shape) == 2:  # if the label tensor has only HxW dimension.
                    label_tensor = label_tensor.unsqueeze(dim=0)

                # label_tensor = Normalize(mean=[0.5], std=[0.5])(label_tensor)

            elif self.input_format in ["png", "PNG", "jpeg", "JPEG", "jpg", "JPG"]:
                transforms = Compose([Lambda(lambda x: self.__to_numpy(x)),
                                      Lambda(lambda x: self.__rotate(x)),
                                      # Lambda(lambda x: self.__pad(x, self.opt.padding_size)),
                                      Lambda(lambda x: self.__random_crop(x)),
                                      Lambda(lambda x: self.__convert_range(x, min=0, max=255)),
                                      ToTensor(),
                                      Normalize(mean=[0.5], std=[0.5])])

                label_array = Image.open(self.label_path_list[index])
                label_tensor = transforms(label_array)

            else:
                NotImplementedError("Please check data_format_input option. It has to be npy or png.")

            if self.target_format in ["fits", "fts", "npy"]:
                if self.target_format in ["fits", "fts"]:
                    target_array = np.array(fits.open(self.target_path_list[index]))
                else:
                    target_array = np.load(self.target_path_list[index], allow_pickle=True)

                target_array = self.__rotate(target_array)
                # target_array = self.__pad(target_array, self.opt.padding_size)
                target_array = self.__random_crop(target_array)
                target_array = self.__convert_range(target_array,
                                                    min=-self.opt.dynamic_range_target,
                                                    max=2 * self.opt.dynamic_range_target)
                target_tensor = torch.tensor(target_array, dtype=torch.float32)
                target_tensor -= 0.5
                target_tensor = target_tensor / 0.5
                target_tensor = target_tensor.unsqueeze(dim=0)  # Add channel dimension.
                # target_tensor = Normalize(mean=[0.5], std=[0.5])(target_tensor)

            elif self.target_format in ["png", "PNG", "jpeg", "JPEG", "jpg", "JPG"]:
                transforms = Compose([Lambda(lambda x: self.__to_numpy(x)),
                                      Lambda(lambda x: self.__rotate(x)),
                                      # Lambda(lambda x: self.__pad(x, self.opt.padding_size)),
                                      Lambda(lambda x: self.__random_crop(x)),
                                      Lambda(lambda x: self.__convert_range(x, min=0.0, max=255.0)),
                                      ToTensor(),
                                      Normalize(mean=[0.5], std=[0.5])])

                target_array = Image.open(self.target_path_list[index])
                target_tensor = transforms(target_array)

            else:
                NotImplementedError("Please check data_format_target option. It has to be fits, fit, npy, jpeg, jpg or png.")

        else:
            if self.input_format in ["fits", "fts", "npy"]:
                if self.input_format in ["fits", "fts"]:
                    label_array = np.array(fits.open(self.label_path_list[index]))
                else:
                    label_array = np.load(self.label_path_list[index], allow_pickle=True)

                label_array = self.__convert_range(label_array,
                                                   min=-self.opt.dynamic_range_input,
                                                   max=2 * self.opt.dynamic_range_input)
                label_tensor = torch.tensor(label_array)
                label_tensor -= 0.5
                label_tensor = label_tensor / 0.5
                label_tensor = label_tensor.unsqueeze(dim=0)

                # label_tensor = Normalize(mean=[0.5], std=[0.5])(label_tensor)

            elif self.input_format in ["png", "PNG", "jpeg", "JPEG", "jpg", "JPG"]:
                transforms = Compose([Lambda(lambda x: self.__to_numpy(x)),
                                      Lambda(lambda x: self.__convert_range(x, min=0, max=255)),
                                      ToTensor(),
                                      Normalize(mean=[0.5], std=[0.5])])

                label_array = Image.open(self.label_path_list[index])
                label_tensor = transforms(label_array)

            else:
                NotImplementedError("Please check data_format option. It has to be npy or png.")

            if self.target_format in ["fits", "fts", "npy"]:
                if self.target_format in ["fits", "fts"]:
                    target_array = np.array(fits.open(self.target_path_list[index]))
                else:
                    target_array = np.load(self.target_path_list[index], allow_pickle=True)

                target_array = self.__convert_range(target_array,
                                                    min=-self.opt.dynamic_range_target,
                                                    max=2 * self.opt.dynamic_range_target)
                target_tensor = torch.tensor(target_array, dtype=torch.float32)
                target_tensor -= 0.5
                target_tensor = target_tensor / 0.5
                target_tensor = target_tensor.unsqueeze(dim=0)  # Add channel dimension.
                # target_tensor = Normalize(mean=[0.5], std=[0.5])(target_tensor)

            elif self.target_format in ["png", "PNG", "jpeg", "JPEG", "jpg", "JPG"]:
                transforms = Compose([Lambda(lambda x: self.__to_numpy(x)),
                                      Lambda(lambda x: self.__convert_range(x, min=0, max=255)),
                                      ToTensor(),
                                      Normalize(mean=[0.5], std=[0.5])])

                target_array = Image.open(self.target_path_list[index])
                target_tensor = transforms(target_array)

            else:
                NotImplementedError("Please check data_format option. It has to be fits, fit, npy, jpeg, jpg or png.")

        return label_tensor, target_tensor, splitext(split(self.label_path_list[index])[-1])[0], \
               splitext(split(self.target_path_list[index])[-1])[0]

    def __random_crop(self, x):
        x = np.array(x)
        x = x[self.offset_x: self.offset_x + 1024, self.offset_y: self.offset_y + 1024]
        return x

    @staticmethod
    def __convert_range(x, min, max):
        x -= min
        x = x / max
        return x

    # @staticmethod
    # def __pad(x, padding_size):
    #     if type(padding_size) == int:
    #         padding_size = ((padding_size, padding_size), (padding_size, padding_size), (padding_size, padding_size))
    #     return np.pad(x, pad_width=padding_size, mode="constant", constant_values=0)

    def __rotate(self, x):
        return rotate(x, self.angle, reshape=False)

    @staticmethod
    def __to_numpy(x):
        return np.array(x, dtype=np.float32)

    def __len__(self):
        return len(self.label_path_list)
