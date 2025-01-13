#!/usr/bin/env python3

import cv2
from dataclasses import dataclass, asdict
from typing import Any, Type


@dataclass
class IMTopic:
    """
    Configuration for an image topic.
    """
    name: str
    width: int = 640
    height: int = 480
    top: int = 0
    bot: int = 0
    right: int = 0
    left: int = 0
    dtype: str = "bgr8"
    flip: bool = False
    info_name: str = None
    is_python_node: bool = False

    def process_image(self, img):
        # Check for overcrop conditions
        assert self.bot + self.top <= img.shape[0], "Overcrop! bot + top crop >= image height!"
        assert self.right + self.left <= img.shape[1], "Overcrop! right + left crop >= image width!"

        # If bot or right is negative, set to value that crops the entire image
        bot = self.bot if self.bot > 0 else -(img.shape[0] + 10)
        right = self.right if self.right > 0 else -(img.shape[1] + 10)

        # Crop image
        img = img[self.top:-bot, self.left:-right]

        # Flip image if necessary
        if self.flip:
            img = img[::-1, ::-1]

        # Resize image if necessary
        if (self.height, self.width) != img.shape[:2]:
            return cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return img

    @classmethod
    def from_dict(cls: Type[Any], data: dict) -> Any:
        return cls(**data)

    def to_dict(self):
        return asdict(self)
