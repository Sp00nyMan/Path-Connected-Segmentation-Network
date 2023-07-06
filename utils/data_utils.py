import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple


class ImageDataset(Dataset):
    def __init__(self, image_original: str | np.ndarray,
                 image_fore: str | np.ndarray,
                 image_back: str | np.ndarray,
                 eps: float = 3e-8, train: bool = True, ignore_rgb: bool = False) -> None:
        image_original = ImageDataset.preprocess_image(image_original)
        if train:
            image_fore = ImageDataset.preprocess_image(image_fore)
            image_back = ImageDataset.preprocess_image(image_back)

            self.mask_fore = image_fore.mean(axis=-1) >= 1 - eps
            self.mask_back = image_back.mean(axis=-1) >= 1 - eps
        else:
            self.mask_fore = np.ones(image_original.shape[:2])
            self.mask_back = np.zeros_like(self.mask_fore)

        foreground_features = ImageDataset.extract_from_mask(image_original, self.mask_fore)
        background_features = ImageDataset.extract_from_mask(image_original, self.mask_back)

        f_labels = np.zeros((len(foreground_features), 1))
        b_labels = np.ones((len(background_features), 1))

        self.data = np.vstack((foreground_features, background_features), dtype="float32")
        self.labels = np.concatenate((f_labels, b_labels), dtype="float32")
        self.rgb = not ignore_rgb

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        return (self.data[index] 
                if self.rgb else self.data[index][:, :2]), self.labels[index]

    @staticmethod
    def preprocess_image(image: str | np.ndarray):
        if isinstance(image, str):
            image = np.array(Image.open(image), dtype=float)
        if len(image.shape) > 3 or image.shape[-1] not in (3, 4):
            raise ValueError(
                f"An image of shape {image.shape} cannot be processed")
        if np.max(image) > 1:
            image /= 255.
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        return image

    @staticmethod
    def extract_from_mask(image: np.ndarray, mask: np.ndarray):
        indices = np.nonzero(mask)
        data = np.zeros((5, len(indices[0])))

        # Normalized Y coordinate of the pixels
        data[0] = indices[0] / image.shape[0]
        # Normalized X coordinate of the pixels
        data[1] = indices[1] / image.shape[1]

        data[2:] = image[indices].T  # RGB values of the pixels

        return data.T
