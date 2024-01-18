import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def create_trainer(args):
    logger = TensorBoardLogger(
        save_dir=f"lightning_logs",
        name=args.log_dir,
        default_hp_metric=False,
        log_graph=True,
    )
    # 設置模型檢查點回調
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename="ckpt_{epoch:02d}_{val/acc:.4f}",
        auto_insert_metric_name=False,
    )

    # 設置學習率監控器
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 計算梯度累積的批次數
    accumulate_grad_batches = args.batch_size // args.min_batch_size
    print(f"Accumulate Grad Batches: {accumulate_grad_batches}")

    # 創建並返回訓練器
    return L.Trainer(
        max_epochs=args.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        devices=args.devices,
        accelerator=args.accelerator,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )
