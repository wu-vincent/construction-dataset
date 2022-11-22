import os.path
from typing import Optional, Callable, Tuple, Any

import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset


class Luo2020(VisionDataset):
    """`Luo2020
    <https://doi.org/10.1016/j.autcon.2019.103016>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``luo2020/equipment_pose_dataset/images/``
            and  ``luo2020/equipment_pose_dataset/labels/`` exist.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        super(Luo2020, self).__init__(root, transforms=transforms, transform=transform,
                                      target_transform=target_transform)

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found. "
                f"You can download it from "
                f"https://hkustbimlab.github.io/"
            )

        self.labels = pd.read_csv(os.path.join(self.labels_dir, "labels.csv"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_id = self.labels.iloc[index, 0]
        image = Image.open(os.path.join(self.images_dir, image_id))
        target = self.labels.iloc[index, 1:].to_dict()

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.images_dir, self.labels_dir]
        return all(os.path.isdir(folder) for folder in folders)

    @property
    def images_dir(self):
        return os.path.join(self.root, self.__class__.__name__.lower(), "equipment_pose_dataset", "images")

    @property
    def labels_dir(self):
        return os.path.join(self.root, self.__class__.__name__.lower(), "equipment_pose_dataset", "labels")
