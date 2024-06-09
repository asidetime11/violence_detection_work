from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from model import *
from dataset import CustomDataModule
import argparse
from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser(description="Violence Test")
    # version : model版本
    parser.add_argument("--version", type=int, default=0, help="Model version")
    parser.add_argument("--bs", type=int, default=4, help="batch size")

    return parser

def main(args):
    gpu_id = [1]
    version =  args.version
    batch_size = args.bs
    log_name = get_log_name(version)
    logger = TensorBoardLogger("test_logs", name=log_name)
    data_module = CustomDataModule(batch_size=batch_size)

    ckpt_path = get_checkpoint_dir(version)
    if version == 0:
        model = ViolenceClassifier_res18.load_from_checkpoint(ckpt_path)
    elif version == 1:
        model = ViolenceClassifier_res34.load_from_checkpoint(ckpt_path)
    trainer = Trainer(accelerator='gpu', devices=gpu_id)
    trainer.test(model, data_module) 

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
