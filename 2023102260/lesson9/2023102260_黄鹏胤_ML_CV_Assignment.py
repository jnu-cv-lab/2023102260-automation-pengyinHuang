import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ====================== 固定路径 ======================
save_dir = ".venv-basic/picture/lesson9"

# ====================== 数据加载（和上次一样） ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
classes = ['0','1','2','3','4','5','6','7','8','9']

# ====================== 任务1：复用你上次的 CNN 模型！ ======================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*5*5, 64)
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

# ====================== 训练函数（统一使用） ======================
def train(model, optimizer, epochs=5):
    criterion = nn.CrossEntropyLoss()
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0,0,0
        for img, lab in train_loader:
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            _, pred = out.max(1)
            t_total += lab.size(0)
            t_correct += pred.eq(lab).sum().item()

        model.eval()
        v_loss, v_correct, v_total = 0,0,0
        with torch.no_grad():
            for img, lab in val_loader:
                out = model(img)
                loss = criterion(out, lab)
                v_loss += loss.item()
                _, pred = out.max(1)
                v_total += lab.size(0)
                v_correct += pred.eq(lab).sum().item()

        tl = t_loss/len(train_loader)
        ta = 100*t_correct/t_total
        vl = v_loss/len(val_loader)
        va = 100*v_correct/v_total

        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)
        print(f"Epoch {epoch+1} | TrainLoss {tl:.4f} | TrainAcc {ta:.2f}% | ValLoss {vl:.4f} | ValAcc {va:.2f}%")

    # 测试集准确率
    test_correct, test_total = 0,0
    with torch.no_grad():
        for img, lab in test_loader:
            out = model(img)
            _, pred = out.max(1)
            test_total += lab.size(0)
            test_correct += pred.eq(lab).sum().item()
    test_acc = 100 * test_correct / test_total
    print(f"Final Test Acc: {test_acc:.2f}%")
    return train_losses, train_accs, val_losses, val_accs, test_acc

# ====================== 任务2：三种优化器对比 ======================
print("\n===== SGD =====")
model_sgd = CNN()
opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
sgd_log = train(model_sgd, opt_sgd)

print("\n===== SGD + Momentum =====")
model_sgdm = CNN()
opt_sgdm = optim.SGD(model_sgdm.parameters(), lr=0.01, momentum=0.9)
sgdm_log = train(model_sgdm, opt_sgdm)

print("\n===== Adam =====")
model_adam = CNN()
opt_adam = optim.Adam(model_adam.parameters(), lr=0.001)
adam_log = train(model_adam, opt_adam)

# 画图对比
epochs = 5
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1,epochs+1), sgd_log[0], label='SGD')
plt.plot(range(1,epochs+1), sgdm_log[0], label='SGD+M')
plt.plot(range(1,epochs+1), adam_log[0], label='Adam')
plt.title('Train Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1,epochs+1), sgd_log[1], label='SGD')
plt.plot(range(1,epochs+1), sgdm_log[1], label='SGD+M')
plt.plot(range(1,epochs+1), adam_log[1], label='Adam')
plt.title('Train Acc')
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/optimizer_compare.png")
plt.close()

# ====================== 任务3：学习率对比（固定 Adam） ======================
lrs = [0.1, 0.01, 0.001]
lr_logs = []
for lr in lrs:
    print(f"\n===== LR = {lr} =====")
    model = CNN()
    opt = optim.Adam(model.parameters(), lr=lr)
    log = train(model, opt, epochs=3)
    lr_logs.append(log)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i,lr in enumerate(lrs):
    plt.plot(range(1,4), lr_logs[i][0], label=f'lr={lr}')
plt.title('Loss by LR')
plt.legend()

plt.subplot(1,2,2)
for i,lr in enumerate(lrs):
    plt.plot(range(1,4), lr_logs[i][1], label=f'lr={lr}')
plt.title('Acc by LR')
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/lr_compare.png")
plt.close()

# ====================== 任务4：第一层卷积核可视化 ======================
kernels = model_adam.conv1.weight.data
plt.figure(figsize=(10,5))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(kernels[i][0].cpu(), cmap='gray')
    plt.axis('off')
plt.suptitle('Conv1 Kernels')
plt.savefig(f"{save_dir}/kernels.png")
plt.close()

# ====================== 任务5：Feature Map 可视化 ======================
img, lab = next(iter(test_loader))
img = img[:1]
feat = model_adam.relu1(model_adam.conv1(img))
feat = feat[0].detach().cpu()

plt.figure(figsize=(10,5))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(feat[i], cmap='gray')
plt.suptitle('Feature Maps (Conv1 Output)')
plt.savefig(f"{save_dir}/feature_maps.png")
plt.close()

# ====================== 任务6：错误样本展示 ======================
errors_img = []
errors_true = []
errors_pred = []

model_adam.eval()
with torch.no_grad():
    for img, lab in test_loader:
        out = model_adam(img)
        _, pred = out.max(1)
        for i in range(len(img)):
            if pred[i] != lab[i]:
                errors_img.append(img[i])
                errors_true.append(lab[i].item())
                errors_pred.append(pred[i].item())
            if len(errors_img) >= 8: break
        if len(errors_img) >=8: break

plt.figure(figsize=(12,4))
for i in range(8):
    plt.subplot(1,8,i+1)
    plt.imshow(errors_img[i][0], cmap='gray')
    plt.title(f"T:{errors_true[i]}\nP:{errors_pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{save_dir}/error_samples.png")
plt.close()

# ====================== 任务7：混淆矩阵 ======================
all_pred = []
all_true = []
with torch.no_grad():
    for img, lab in test_loader:
        out = model_adam(img)
        _, pred = out.max(1)
        all_pred.extend(pred.cpu().numpy())
        all_true.extend(lab.cpu().numpy())

cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Pred')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f"{save_dir}/confusion_matrix.png")
plt.close()

print("\n✅ 全部任务完成！图片已保存到：", save_dir)