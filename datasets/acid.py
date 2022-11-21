import os.path
from typing import Optional, Callable, Tuple, Any

from PIL import Image
from torchvision.datasets import VisionDataset


class ACID(VisionDataset):
    """`ACID
    <https://doi.org/10.1061/(asce)cp.1943-5487.0000945>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``ACID/ACID_Annotations/``
            and  ``ACID/ACID_Images/`` exist.
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
        super(ACID, self).__init__(root, transforms=transforms, transform=transform,
                                   target_transform=target_transform)

        self.images = []
        self.targets = []

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found. "
                f"You can download it from "
                f"https://www.acidb.ca/"
            )

        for img_file in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, img_file))
            self.targets.append(os.path.join(self.labels_dir, f"{img_file.split('.')[0]}.xml"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index])
        target = []
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.targets[index])
        root = tree.getroot()
        for obj in root.findall('object'):
            # noinspection PyTypeChecker
            target.append(
                {
                    "name": str(obj.findtext("name", default="N/A")),
                    "pose": str(obj.findtext("pose", default="N/A")),
                    "truncated": bool(int(obj.findtext("truncated", default="0"))),
                    "difficult": bool(int(obj.findtext("difficult", default="0"))),
                    "bbox": [int(x.text) for x in obj.find("bndbox")],
                }
            )

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.images_dir, self.labels_dir]
        return all(os.path.isdir(folder) for folder in folders)

    @property
    def images_dir(self):
        return os.path.join(self.root, self.__class__.__name__, "ACID_Images")

    @property
    def labels_dir(self):
        return os.path.join(self.root, self.__class__.__name__, "ACID_Annotations")
