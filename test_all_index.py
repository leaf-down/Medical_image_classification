import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score)
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

# 自己改模型导入
from MedSSD_kan.MedSSD_kan import VSSM as medmamba  # import model

# path
dataset_dir = "/app/RetinalOCT_Dataset/test"
model_path = "/app/models/"
model_name = "ssd_2kan_Net_retinal.pth"

# data preprocessing
batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale images to 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
test_num = len(test_dataset)
print("using {} images for test.".format(test_num))

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# Load model
model = medmamba(num_classes=8)  # num_classes is the number of classes the model needs to classify
model.load_state_dict(torch.load(model_path+model_name))
model = model.to(device)

# Test
model.eval()
y_true = []
y_pred = []
y_prob = []
with torch.no_grad():
    test_bar = tqdm(test_loader, file=sys.stdout)
    for test_data in test_bar:
        test_images, test_labels = test_data
        outputs = model(test_images.to(device))
        probs = torch.softmax(outputs, dim=1)  # 获取概率输出
        predict_y = torch.max(outputs, dim=1)[1]

        y_true.extend(test_labels.cpu().numpy())
        y_pred.extend(predict_y.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

# Convert to numpy arrays
y_true = torch.tensor(y_true).numpy()
y_pred = torch.tensor(y_pred).numpy()
y_prob = torch.tensor(y_prob)[:, 1].numpy()  # 假设类别1是正类

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
specificity = recall_score(y_true, y_pred, pos_label=0, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
auc = roc_auc_score(y_true, y_prob, multi_class='ovr')

# Parameter size
param_size = sum(p.numel() for p in model.parameters())

# Save results
results = pd.DataFrame({
    'Model': [model_name],
    'Dataset': ['RetinalOCT_Dataset'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Sensitivity': [recall],
    'Specificity': [specificity],
    'F1-Score': [f1],
    'AUC': [auc],
    'Parameter Size': [param_size]
})

filename = "/app/models/csv/evaluation_{}_RetinalOCT_Dataset.csv".format(model_name)
results.to_csv(filename, index=False)
print(f"Results saved to {filename}")
