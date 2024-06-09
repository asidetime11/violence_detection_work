import torch
from pytorch_lightning.loggers import TensorBoardLogger
import time
import os
from PIL import Image
from model import *
from dataset_for_fgsm import *
from torchvision import transforms

gpu_id = [1]
batch_size = 1024
log_name = "resnet18_pretrain"

data_module = CustomDataModule(batch_size=batch_size)
ckpt_path = "checkpoints/resnet18_pretrain_test-epoch=03-val_loss=0.21.ckpt"
logger = TensorBoardLogger("test_logs", name=log_name)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 由于gpu内存不足,放在cpu上运行
device ='cpu'

model = ViolenceClassifier_res18.load_from_checkpoint(ckpt_path).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)

output_folder_path = "violence_224/test2_new"

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def denorm(batch, mean=[0.1307], std=[0.3081]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def FGSM_test(net,eval_loader,device,criterion, output_folder_path, ep = 0.1):
    net.eval()
    test_loss, correct, total = 0,0,0
    b_time = time.time()
    for batch_id, sample in enumerate(eval_loader):
        x, target = next(iter(sample)) # 使用next获取下一批次的x和target
        x, target = x.to(device), target.to(device)
        x.requires_grad = True

        y = net(x)
        _,predicted = y.max(1)

        if predicted.detach().cpu().numpy()[0] != target.detach().cpu().numpy()[0]:
            continue
            
        loss = criterion(y, target).to(device)
        net.zero_grad()
        loss.backward()
        data_grad = x.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(x)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, ep, data_grad)

        # Reapply normalization
        adv_x = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        adv_y = net(adv_x)
        
        loss= criterion(adv_y,target)
        test_loss+=loss.item()
        _,adv_predicted = adv_y.max(1)
        correct += torch.eq(target,adv_predicted).float().sum().item()
        
        total+=target.size(0)

        # Save the adversarial images
        for idx in range(x.size(0)):
            adv_image = adv_x[idx].cpu().detach()
            adv_image = (adv_image * 0.5 + 0.5) * 255  # Denormalize
            adv_image = adv_image.permute(1, 2, 0).numpy().astype('uint8')

            image_name = f"{batch_id * batch_size + idx}.jpg"
            # 将label加入图片路径中,用于test
            label_value = str(target[idx].item())
            image_path = os.path.join(output_folder_path, f"{label_value}_{image_name}")
            # 检查路径是否存在，如果不存在则创建
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.fromarray(adv_image).save(image_path)

    acc = 100. * correct / total
    t_time = time.time()
    print("Test acc:{0:.3f}% with epsilon is {1:.2f} , Loss: {2:.3f}".format(acc,ep,test_loss))
    print("Consuming time:{:.3f} s".format(t_time-b_time))

FGSM_test(model, data_module.test_dataloader(), device, criterion, output_folder_path, 0.05)
