from model import *
import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import os
from utils import *

class ViolenceClass:
    def __init__(self, version = 0):
        ckpt_path = get_checkpoint_dir(version)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if version == 0:
            self.model = ViolenceClassifier_res18.load_from_checkpoint(ckpt_path, map_location=self.device)
        elif version == 1:
            self.model = ViolenceClassifier_res34.load_from_checkpoint(ckpt_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 定义预处理变换
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_image(self,image_path):
        # 加载本地图片并进行预处理
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        return image
        
    def classify(self, imgs : torch.Tensor) -> list:
        # 图像分类
        imgs = imgs.to(self.device)
        with torch.no_grad():
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().tolist()

# 采用命令行格式进行调用
def get_args_parser():
    parser = argparse.ArgumentParser(description="Violence Classification")
    # version : model版本
    parser.add_argument("--version", type=int, default=0, help="Model version")
    # dir : 图片文件夹路径,可以多个
    parser.add_argument("--dir", type=str, nargs='+', required=True, help="Directory or list of directories containing images")

    return parser

def main(args):
    # 加载模型
    classifier = ViolenceClass(version=args.version)

    # 加载本地图片并进行预处理
    image_paths = []
    for directory in args.dir:
        if os.path.isdir(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
        elif os.path.isfile(directory) and directory.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(directory)

    if not image_paths:
        print("No images found in the specified directories.")

        return

    images = [classifier.load_image(image_path) for image_path in image_paths]
    images = torch.stack(images)  # 将图片堆叠成一个batch

    # 分类
    predictions = classifier.classify(images)
    for img_path, pred in zip(image_paths, predictions):
        print(f"Image: {img_path}, Prediction: {pred}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

