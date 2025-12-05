import os
import torch.nn.functional as F
import random
import re
import torch
from torch.utils.data import Dataset
from PIL import Image


class GANDIV2KDataLoader(Dataset):
    """
    A PyTorch `Dataset` class for DIV2K related challenges

    This dataset supports three modes of operation:
    - **'train'**: Loads the training dataset
    - **'val'** / **'test'**: Loads the low resolution dataset and their high resolution 
    counterparts from a given file.

    Attributes
    ----------
    root_dir_lr : str
        Path to the directory containing low resolution images.
    root_dir_hr : str
        Path to the directory containing high resolution images.
    transform : callable, optional
        A torchvision-compatible transform applied to each image.
    mode : str
        One of {'train', 'val', 'test'} indicating dataset behavior.
    batch_size : int
        In training mode, the number of samples per epoch
    scale : int
        In training mode, the difference in scale between the hr image and lr image
    patch_size : int
        In training mode, the size of the random cropping window
    """

    def __init__(self, root_dir_lr, root_dir_hr, transformLr=None, transformHr=None, mode='train', 
                 batch_size=None, scale = 8, patch_size = 64):
        """
        Initialize the ProgressionDataset.

        Parameters
        ----------
        root_dir_lr : str
            Path to the directory containing low resolution images.
        root_dir_hr : str
            Path to the directory containing high resolution images.
        transform : callable, optional
            A torchvision-compatible transform applied to each image.
        mode : str
            One of {'train', 'val', 'test'} indicating dataset behavior.
        batch_size: int
            In training mode, the number of samples per epoch
        scale : int
            In training mode, the difference in scale between the hr image and lr image
        patch_size : int
            In training mode, the size of the random cropping window
        """
        self.transformLr = transformLr
        self.transformHr = transformHr
        self.batch_size = batch_size
        self.mode = mode
        self.root_dir_lr = root_dir_lr
        self.root_dir_hr = root_dir_hr
        self.scale = scale
        self.patch_size = patch_size

        # Order the two folders
        self.lr_image_files = sorted([f for f in os.listdir(
            root_dir_lr) if f.lower().endswith('.png')])
        self.hr_image_files = sorted([f for f in os.listdir(
            root_dir_hr) if f.lower().endswith('.png')])

        self.lr_image_files.sort()
        self.hr_image_files.sort()

        print(f"Found {len(self.lr_image_files)} images for {self.mode} task.")
        print(f"Found {len(self.hr_image_files)} images for {self.mode} task.")

    def _generate_random_crop(self, lr_img, hr_img):
        """
        Randomly selects a cropping position and crops the same for both the 
        high resolution and low resolution image.

        Returns
        -------
        tuple
            (cropped_img_lr, cropped_img_hr)
        """
        # Select crop pos
        w, h = lr_img.size
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        # crop lr img
        lr_crop = lr_img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # Match crop for hr img
        file_scale = 8

        x_hr = x * file_scale
        y_hr = y * file_scale
        hr_crop = hr_img.crop((
            x_hr,
            y_hr,
            x_hr + (self.patch_size * file_scale),
            y_hr + (self.patch_size * file_scale),
        ))
        return lr_crop, hr_crop

    def _generate_pair(self):
        """
        Randomly select a training image and its high resolution counterpart.

        Returns
        -------
        tuple
            (img_path_lr, img_path_hr)
        """
        lr_img_path = random.choice(self.lr_image_files)
        code = lr_img_path[:4]
        for item in self.hr_image_files:
            if item[:4] == code:
                hr_img_path = item
                break

        return lr_img_path, hr_img_path

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            `batch_size` in training mode, or number of fixed pairs otherwise.
        """
        return self.batch_size if self.mode == 'train' else len(self.lr_image_files)

    def __getitem__(self, idx=None):
        """
        Retrieve a dataset sample at the given index.

        Parameters
        ----------
        idx : int, optional
            Sample index (used only in 'val'/'test' mode), defaults to None
            assuming a 'train' mode.

        Returns
        -------
        tuple
            (img_a, img_b, label)
            where `img_a` and `img_b` are transformed tensors,
            and `label` is a torch.LongTensor indicating the pair type.
        """
        if self.mode == 'train':
            img_lr_path, img_hr_path = self._generate_pair()
        else:
            if idx is None:
                raise ValueError(
                    "In 'val'/'test' mode, 'idx' must be provided.")
            img_lr_path = self.lr_image_files[idx]
            img_hr_path = self.hr_image_files[idx]

        img_lr = Image.open(self.root_dir_lr + "/" + img_lr_path).convert("RGB")
        img_hr = Image.open(self.root_dir_hr + "/" + img_hr_path).convert("RGB")

        # Random cropping
        if self.mode == 'train':
            img_lr, img_hr = self._generate_random_crop(img_lr, img_hr)

        if self.transformLr:
            img_lr = self.transformLr(img_lr)
        if self.transformHr:
            img_hr = self.transformHr(img_hr)

        if self.scale == 16:
             img_lr = F.interpolate( img_lr.unsqueeze(0), scale_factor=0.5, mode="bicubic", align_corners=False).squeeze(0)

        return img_lr, img_hr
