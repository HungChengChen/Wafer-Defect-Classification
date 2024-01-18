import lightning as L
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics.classification import BinaryAccuracy

from models.model_factory import create_timm_model


# Define a PyTorch Lightning module for binary classification.
class BinaryModule(L.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # Hyperparameters
        self.epochs = args.epochs
        self.lr = args.lr
        self.eta_min = args.eta_min
        self.batch_size = args.min_batch_size
        self.num_workers = args.num_workers

        # Create the model; raise error if not binary classification
        if args.num_classes != 2:
            raise ValueError("num_classes must be 2 for binary classification.")
        self.model = create_timm_model(
            model_name=args.model_name,
            num_classes=1,
            use_pretrained=args.use_pretrained,
        )
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        # Metrics
        self.train_acc = BinaryAccuracy(threshold=0.5)
        self.val_acc = BinaryAccuracy(threshold=0.5)

        # self.example_input_array = torch.randn(1, 3, 224, 224)

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        labels = labels.float()
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        # Log training loss
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        # Update training accuracy
        self.train_acc.update(logits, labels)
        return loss

    def on_train_epoch_end(self):
        # Log training accuracy at the end of each epoch
        self.log(
            "train/acc",
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        labels = labels.float()
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        # Log validation loss
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        # Update validation accuracy
        self.val_acc.update(logits, labels)

    def on_validation_epoch_end(self):
        # Log validation accuracy at the end of each epoch
        self.log(
            "val/acc",
            self.val_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.val_acc.reset()

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # Configure scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.eta_min, last_epoch=-1
        )
        return [optimizer], [scheduler]
