import os
import random
from typing import List, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image


class Mask:
    def __init__(
        self,
        config,
        mask_size: int = 299,
        prob_use_mask: float = 0,
        rand_mask_colors: Union[None, List] = None,
        test_mask_color: Union[None, str] = None,
        CMUNet_path: Union[None, str] = None,
        cuda_device: int = 0,
    ):
        self.config = config
        self.mask_size = mask_size
        self.prob_use_mask = prob_use_mask
        self.rand_mask_colors = rand_mask_colors
        self.test_mask_color = test_mask_color
        self.cuda_device = cuda_device
      

    def apply_mask_color(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        mask_color: Union[None, str, List] = None,
    ) -> np.ndarray:
        try:
            if mask_color is None:
                mask_color = random.choice(self.rand_mask_colors)
            mask_bg_color = [0, 0, 0]
            if type(mask_color) == str:
                if mask_color == "avg":
                    mask_bg_color = np.average(image[mask == 255], axis=0)
                if mask_color == "rand_avg":
                    mask_bg_color = np.random.normal(
                        np.mean(image[mask == 255], axis=0),
                        np.std(image[mask == 255], axis=0),
                    )
                if mask_color == "min":
                    mask_bg_color = np.min(image[mask == 255], axis=0)
                if mask_color == "max":
                    mask_bg_color = np.min(image[mask == 255], axis=0)
            if type(mask_color) == list:
                mask_bg_color = mask_color
            image[mask == 255] = mask_bg_color
        except Exception as e:
            err = e
        return image

    def load_mask(self, image: np.ndarray, current_sample: dict, transform) -> np.ndarray:
        mask_path = current_sample["Mask_Path"]


        mask = cv2.imread(mask_path)
        # print(mask_path, mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        mask = np.array(mask)
        return mask

    def apply_mask(self, image: np.ndarray, mask: np.ndarray, mode: str = "train") -> np.ndarray:
        if mode == "test":
            image = self.apply_mask_color(image, mask, mask_color=self.test_mask_color)
        else:
            if (1 - self.prob_use_mask) <= random.random():
                image = self.apply_mask_color(image, mask)
               

        return image