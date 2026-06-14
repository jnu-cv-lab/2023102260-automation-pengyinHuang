import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

# 导入模型
import sys
sys.path.append(os.path.dirname(__file__))
from model import SkeletonTransformer

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 【统一成果路径】
OUTPUT_ROOT = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson11"
BATCH_SIZE = 8
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 8   # 早停阈值
# 模型保存路径
SAVE_MODEL_PATH = os.path.join(OUTPUT_ROOT, "best_model.pth")

# ===================== 自定义数据集 =====================
class SkeletonDataset(Dataset):
    def __init__(self, data_npy, label_npy):
        self.data = np.load(data_npy)
        self.labels = np.load(label_npy)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        feat = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label

# ===================== 训练&验证函数 =====================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for feats, labels in tqdm(loader, desc="训练中"):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / len(loader)
    avg_acc = correct / total
    return avg_loss, avg_acc

def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, labels in tqdm(loader, desc="验证中"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    avg_acc = correct / total
    return avg_loss, avg_acc

# ===================== 主训练流程 =====================
def main():
    # 加载数据集
    train_data_path = os.path.join(OUTPUT_ROOT, "X_train.npy")
    train_label_path = os.path.join(OUTPUT_ROOT, "y_train.npy")
    test_data_path = os.path.join(OUTPUT_ROOT, "X_test.npy")
    test_label_path = os.path.join(OUTPUT_ROOT, "y_test.npy")
    train_dataset = SkeletonDataset(train_data_path, train_label_path)
    test_dataset = SkeletonDataset(test_data_path, test_label_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型、损失、优化器
    model = SkeletonTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度 + 早停初始化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    best_acc = 0.0
    early_stop_cnt = 0
    best_model = None

    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = val_epoch(model, test_loader, criterion)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"训练 Loss: {train_loss:.4f} | 训练 Acc: {train_acc:.4f}")
        print(f"测试 Loss: {val_loss:.4f} | 测试 Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = deepcopy(model.state_dict())
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= PATIENCE:
                print(f"\n早停触发，最优准确率: {best_acc:.4f}")
                torch.save(best_model, SAVE_MODEL_PATH)
                break

        # 更新学习率
        scheduler.step(val_acc)

    # 绘制训练曲线，保存到指定路径
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.legend()
    curve_path = os.path.join(OUTPUT_ROOT, "train_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"\n📊 训练曲线已保存至: {curve_path}")
    print(f"🏆 最优测试准确率: {best_acc:.4f}")

if __name__ == "__main__":
    main()