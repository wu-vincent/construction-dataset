import json
import os.path
import pickle
import time
from typing import Optional, Callable, Tuple, Any

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg


class MOCS(VisionDataset):
    """`MOCS
    <https://doi.org/10.1016/j.autcon.2020.103482>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MOCS/instances_train/``, ``MOCS/instances_val/``
            and  ``MOCS/instances_test/`` exist.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val``
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
            split: str = "train",
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        self.split = verify_str_arg(split, "split", ["train", "test", "val"])

        super(MOCS, self).__init__(root, transforms=transforms, transform=transform,
                                   target_transform=target_transform)

        self.images = []
        self.targets = {}

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found. "
                f"You can download it from "
                f"http://www.anlab340.com/Archives/IndexArctype/index/t_id/17.html/"
            )

        file_name = f"instances_{self.split}" if self.split != "test" else f"image_info_test"
        self.json_file = os.path.join(self.labels_dir, f"{file_name}.json")
        self.pkl_file = os.path.join(self.labels_dir, f"{file_name}.pkl")

        tic = time.time()
        try:
            self._load_from_pickle()
        except FileNotFoundError:
            self._load_from_json()
            self._save_to_pickle()
        toc = time.time()
        print(f"MOCS loaded in {(toc - tic) * 10000 // 10}ms")

    def _load_from_json(self):
        with open(self.json_file) as fp:
            data = json.load(fp)
            for image in data["images"]:
                image_id = image["id"]
                file_name = os.path.join(self.images_dir, image["file_name"])
                self.images.append((image_id, file_name))

            categories = {}
            for category in data["categories"]:
                category_id = category["id"]
                category_name = category["name"]
                categories[category_id] = category_name

            if self.split != "test":
                for annotation in data["annotations"]:
                    image_id = annotation["image_id"]
                    parts = []
                    for part in annotation["segmentation"]:
                        parts.append(np.reshape(np.array(part), (-1, 2)))

                    target = {
                        "segmentation": parts,
                        "bbox": annotation["bbox"],
                        "category": categories[int(annotation["category_id"])],
                    }
                    if image_id not in self.targets:
                        self.targets[image_id] = []

                    self.targets[image_id].append(target)

    def _save_to_pickle(self):
        with open(self.pkl_file, 'wb') as handle:
            pickle.dump({
                "images": self.images,
                "targets": self.targets,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_from_pickle(self):
        with open(self.pkl_file, 'rb') as handle:
            data = pickle.load(handle)
            self.images = data["images"]
            self.targets = data["targets"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_id, image_file = self.images[index]
        image = Image.open(image_file)
        target = self.targets[image_id]

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.images_dir, self.labels_dir]
        return all(os.path.isdir(folder) for folder in folders)

    @property
    def images_dir(self):
        return os.path.join(self.root, self.__class__.__name__, f"instances_{self.split}")

    @property
    def labels_dir(self):
        return os.path.join(self.root, self.__class__.__name__)
