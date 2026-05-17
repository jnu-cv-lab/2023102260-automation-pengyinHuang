import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ====================== 固定图片保存路径（你已创建好） ======================
save_dir = ".venv-basic/picture/lesson8"

# ====================== 任务2：加载 MNIST 数据集 ======================
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 训练集 → 训练集 + 验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 类别
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 显示 8 张训练样本
def plot_images(images, labels, preds=None, title="Samples", save_name="img.png"):
    plt.figure(figsize=(12, 4))
    for i in range(8):
        plt.subplot(1, 8, i+1)
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        if preds is None:
            plt.title(f"True: {classes[labels[i]]}")
        else:
            plt.title(f"T:{classes[labels[i]]}\nP:{classes[preds[i]]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{save_name}")
    plt.close()

# 画训练集样本
data_iter = iter(train_loader)
images, labels = next(data_iter)
plot_images(images, labels, title="MNIST 训练集样本", save_name="train_samples.png")

# ====================== 任务3：定义 CNN 模型 ======================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

model = CNN()

# ====================== 任务4：训练设置 ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# 记录曲线
train_losses = []
train_accs = []
val_losses = []
val_accs = []

# ====================== 训练 + 验证（任务4、5） ======================
for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%\n")

# ====================== 任务6：测试模型 ======================
model.eval()
test_loss = 0
correct = 0
total = 0
all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_images.append(images)
        all_labels.append(labels)
        all_preds.append(predicted)

test_loss /= len(test_loader)
test_acc = 100 * correct / total

print("===== 测试集结果 =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# 显示 8 张测试图
images = torch.cat([x for x in all_images])[:8]
labels = torch.cat([x for x in all_labels])[:8]
preds = torch.cat([x for x in all_preds])[:8]
plot_images(images, labels, preds, title="测试集预测结果", save_name="test_predictions.png")

# ====================== 任务7：绘制训练曲线 ======================
# Loss 曲线
plt.figure()
plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, epochs+1), val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{save_dir}/loss_curve.png")
plt.close()

# Acc 曲线
plt.figure()
plt.plot(range(1, epochs+1), train_accs, label="Train Acc")
plt.plot(range(1, epochs+1), val_accs, label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig(f"{save_dir}/acc_curve.png")
plt.close()

print("所有图片已保存到：", save_dir)