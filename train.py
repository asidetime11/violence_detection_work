from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import *
from dataset import CustomDataModule
import argparse
import os
from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser(description="Violence Train")
    # version : model版本
    parser.add_argument("--version", type=int, default=0, help="Model version")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--bs", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=40, help="num epochs")

    return parser

def main(args):
    gpu_id = [1]
    lr = args.lr
    version = args.version
    # 减小batch_size,防止out of memory
    batch_size = args.bs
    # 指定日志名称,verison=0是res18,1是res34
    log_name = get_log_name(version)
    print("{} gpu: {}, batch size: {}, lr: {}".format(log_name, gpu_id, batch_size, lr))
    
    data_module = CustomDataModule(batch_size=batch_size)
    # 设置模型检查点，用于保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=log_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    logger = TensorBoardLogger("train_logs", name=log_name)

    # 实例化训练器
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=gpu_id,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    if version==0:
        model = ViolenceClassifier_res18(learning_rate=lr)
    elif version==1:
        model = ViolenceClassifier_res34(learning_rate=lr)

    # 开始训练
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
