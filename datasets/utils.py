import os
from typing import Any, Callable, Dict, List, Optional, Tuple


def find_classes(
    directory: str, ignore_classes: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Discovers the distinct class folders within a given directory, useful for dataset organization.

    Args:
        directory (str): The directory to search for class folders.
        ignore_classes (Optional[List[str]]): A list of class names to ignore during the search.

    Returns:
        classes (List[str]): List of class names found in the directory (e.g., ['dog', 'cat']).
        class_to_idx (Dict[str, int]): Mapping from class names to their respective indices (e.g., {'dog': 0, 'cat': 1}).
        idx_to_class (Dict[int, str]): Mapping from indices to class names (e.g., {0: 'dog', 1: 'cat'}).

    Raises:
        FileNotFoundError: If no class folders are found in the provided directory.
    """

    # Discover class directories, skipping over any specified in ignore_classes
    classes = [
        entry.name
        for entry in os.scandir(directory)
        if entry.is_dir() and (not ignore_classes or entry.name not in ignore_classes)
    ]

    # Raise an error if no class directories were found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    # Alphabetically sorts the class names.
    classes.sort()  # Example: For ['dog', 'cat']

    # Create a dictionary mapping each class to an index.
    class_to_idx = {
        cls_name: i for i, cls_name in enumerate(classes)
    }  # (e.g., {'dog': 0, 'cat': 1})

    # Create a reverse mapping from indices to class names.
    idx_to_class = {
        i: cls_name for i, cls_name in enumerate(classes)
    }  # (e.g., {0: 'dog', 1: 'cat'})

    # Log a summary of the classes found
    print(f"Found {len(classes)} classes in '{directory}': {', '.join(classes)}")
    return classes, class_to_idx, idx_to_class


def make_dataset(
    directory: str, class_to_idx: Dict[str, int]
) -> List[Tuple[str, str, int]]:
    """
    Creates a dataset by scanning a directory and associating files with their respective classes.

    Args:
        directory (str): The root directory of the dataset.
        class_to_idx (Dict[str, int]): A mapping from class names to indices.

    Returns:
        List[Tuple[str, str, int]]: A list of tuples, each containing a file path and its class index.
    """

    # Expand the user's home directory (~) if used in the path
    directory = os.path.expanduser(directory)

    # Initialize a list to store tuples of (file_path, class_index)
    instances = []

    # Loop over each class name sorted alphabetically
    for target_class in sorted(class_to_idx.keys()):
        # Retrieve the index associated with the class
        class_index = class_to_idx[target_class]
        # Construct the full path to the class directory
        target_dir = os.path.join(directory, target_class)

        # Traverse the directory tree starting from target_dir
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            # Iterate over each file name, sorted alphabetically
            for fname in sorted(fnames):
                # Construct the full path to the file
                path = os.path.join(root, fname)
                # Add a tuple of the file path and its class index to the instances list
                instances.append((path, class_index))

    # Return the list of instances (file paths with corresponding class indices)
    return instances
