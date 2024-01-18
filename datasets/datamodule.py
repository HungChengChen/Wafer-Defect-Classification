from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import lightning as L
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        args,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ):
        super().__init__()
        self.fold_index = 0
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset if val_dataset is not None else train_dataset
        self.test_dataset = test_dataset

        self.seed: int = args.seed
        self.n_splits = args.n_splits
        self.min_batch_size: int = args.min_batch_size
        self.num_workers: int = args.num_workers
        self.sampling: str = args.sampling

        if not hasattr(train_dataset, "targets"):
            raise ValueError(
                "Train dataset must have a 'targets' attribute for stratified splitting."
            )
        self.args.num_classes = len(train_dataset.classes)
        self.args.class_to_idx = train_dataset.class_to_idx
        self.args.idx_to_class = train_dataset.idx_to_class
        # self.args.example_input_array = train_dataset.example_input_array

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            labels = self.train_dataset.targets
            if self.sampling == "kfold":
                splitter = StratifiedKFold(
                    self.n_splits, shuffle=True, random_state=self.seed
                )
                self.splits = [
                    split
                    for split in splitter.split(
                        range(len(self.train_dataset)), y=labels
                    )
                ]
            elif self.sampling == "stratified":
                train_ids, val_ids = train_test_split(
                    range(len(self.train_dataset)),
                    test_size=1.0 / self.n_splits,
                    random_state=self.seed,
                    stratify=labels,
                )
                self.splits = [(train_ids, val_ids)]

    def _create_weighted_sampler(self, dataset: Dataset, ids: List[int]):
        class_counts = Counter([dataset.targets[i] for i in ids])
        weights = [1.0 / class_counts[dataset.targets[i]] for i in ids]
        return WeightedRandomSampler(weights, len(ids), replacement=True)

    def train_dataloader(self) -> DataLoader:
        train_ids = self.splits[self.fold_index][0]
        sampler = self._create_weighted_sampler(self.train_dataset, train_ids)
        return DataLoader(
            Subset(self.train_dataset, train_ids),
            sampler=sampler,
            batch_size=self.min_batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_ids = self.splits[self.fold_index][1]
        return DataLoader(
            Subset(self.val_dataset, val_ids),
            shuffle=False,
            batch_size=self.min_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset, shuffle=False, batch_size=1, num_workers=0
            )
