from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .utils import find_classes, make_dataset


def pil_loader(path: str) -> Image.Image:
    """
    Loads an image from the given file path using PIL and converts it to RGB format.

    Args:
        path (str): The file path to the image.

    Returns:
        Image.Image: The loaded image in RGB format.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")  # Convert the image to RGB format


def default_loader(path: str) -> Any:
    return pil_loader(path)


class ImageFolder(Dataset):
    """
    A generic dataset loader where images are stored in a hierarchical directory structure.
    Each class has its own folder, with images for that class stored within.

    For example, images are expected to be organized in the following structure:
    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/[...]/xxz.png
    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/[...]/asd932_.png

    Args:
        root (string): Root directory path of the dataset.
        image_size (int, optional): The size to which images are resized. Default is 224.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g., `transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        ignore_classes (List[str], optional): Classes to be ignored while loading the dataset.

    Attributes:
        classes (List[str]): List of class names sorted alphabetically.
        class_to_idx (Dict[str, int]): Dict mapping class names to their respective indices.
        idx_to_class (Dict[int, str]): Dict mapping indices to their respective class names.
        imgs (List[Tuple[str, int]]): List of (image path, class_index) tuples.
        targets (List[int]): List of class indices corresponding to each image.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        ignore_classes: Optional[List[str]] = None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.ignore_classes = ignore_classes

        # Load class information and create a dataset
        self.classes, self.class_to_idx, self.idx_to_class = self._find_classes(
            directory=self.root, ignore_classes=self.ignore_classes
        )
        self.num_classes = len(self.classes)
        self.samples = self._make_dataset(
            directory=self.root, class_to_idx=self.class_to_idx
        )
        self.imgs = self.samples
        self.targets = [s[1] for s in self.samples]

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def _find_classes(
        self, directory: str, ignore_classes: Optional[List[str]]
    ) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        """
        Finds and enumerates classes in a dataset directory. This method can be overridden to
        adapt to different dataset structures or to filter specific classes.

        Args:
            directory (str): Path to the root directory of the dataset.

        Returns:
            Tuple containing:
                - List of class names.
                - Dictionary mapping class names to indices.
                - Dictionary mapping indices to class names.
        """
        return find_classes(directory, ignore_classes)

    def _make_dataset(
        self, directory: str, class_to_idx: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        """
        Generates a list of image file paths and their corresponding class indices.

        Args:
            directory (str): Path to the root directory of the dataset.
            class_to_idx (Dict[str, int]): Dictionary mapping class names to indices.

        Returns:
            List of tuples, each containing a file path and a class index.
        """
        return make_dataset(directory, class_to_idx)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Retrieve an image and its corresponding class index by the dataset index.

        Args:
            index (int): Index of the sample in the dataset.

        Returns:
            Tuple containing the image and its class index.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        # Apply the provided transform to the sample
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path
