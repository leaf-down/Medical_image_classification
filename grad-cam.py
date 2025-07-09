import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from MedSSD_kan.MedSSD_kan import VSSM as module

# pth path
model_path = "/app/models/ssd_kan_Net.pth"

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# === 运行 Grad-CAM ===
model = module(num_classes=8)  # 替换为你的模型
model.load_state_dict(torch.load(model_path))
model = model.to(device)  # 确保模型在正确的设备上
model.eval()  # 设置为评估模式

# 假设 kans 是你的 KAN 层对象
for name, param in model.kans.named_parameters():
    if 'grid' in name or 'mask' in name:
        param.requires_grad_(True)  # 强制启用梯度

print("更新后参数状态:")
for name, param in model.kans.named_parameters():
    print(f"{name} {param.requires_grad}")
