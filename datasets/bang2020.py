import os.path
from typing import Optional, Callable, Any, Tuple

import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset


class Bang2020(VisionDataset):
    """`Bang2020
    <https://doi.org/10.1016/j.autcon.2020.103116>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``bang2020/UVAData/``
            and  ``bang2020/via_region_data_final`` exist.
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
        super(Bang2020, self).__init__(root, transforms=transforms, transform=transform,
                                       target_transform=target_transform)

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found. "
                f"You can download it from "
                f"https://data.mendeley.com/datasets/4h68fmktwh/1"
            )

        self.region_data = pd.read_json(os.path.join(self.labels_dir, "via_region_data_final.json"), orient='index')

    def __len__(self):
        return len(self.region_data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        dat = self.region_data.iloc[index]
        image_id = dat["filename"]
        image = Image.open(os.path.join(self.images_dir, image_id))

        target = []
        regions: dict = dat["regions"]
        for (k, v) in regions.items():
            shape_attributes = v["shape_attributes"]
            assert shape_attributes["name"] == "rect"
            x = int(shape_attributes["x"])
            y = int(shape_attributes["y"])
            w = int(shape_attributes["width"])
            h = int(shape_attributes["height"])

            phrase = str(v["region_attributes"]["phrase"])

            target.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "phrase": phrase,
            })

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.labels_dir, self.images_dir]
        return all(os.path.isdir(folder) for folder in folders)

    @property
    def labels_dir(self):
        return os.path.join(self.root, self.__class__.__name__.lower())

    @property
    def images_dir(self):
        return os.path.join(self.root, self.__class__.__name__.lower(), "UAVData")
