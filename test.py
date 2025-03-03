import os
import sys
import torch
from tqdm import tqdm

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

# 自己改模型导入
from MedMamba import VSSM as medmamba  # import model

# path
dataset_dir = "/app/RetinalOCT_Dataset/test"
model_path = "/app/models/MedmambaNet.pth"

# data preprocessing
batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale images to 3 channels
    transforms.Resize((224, 224)),  # Resize shorter side to 256, keeping aspect ratio
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization is consistent with RGB images
])
test_dataset = datasets.ImageFolder(root=dataset_dir,
                                    transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size, shuffle=False,
                                          num_workers=nw)
test_num = len(test_dataset)
print("using {} images for test.".format(test_num))

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# Load model
model = medmamba(num_classes=8)  # num_classes is the number of classes the model needs to classify
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# test
model.eval()
acc = 0.0  # accumulate accurate number / epoch
with torch.no_grad():
    test_bar = tqdm(test_loader, file=sys.stdout)
    for test_data in test_bar:
        test_images, test_labels = test_data
        outputs = model(test_images.to(device))
        predict_y = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

test_accurate = acc / test_num

print(test_accurate)


