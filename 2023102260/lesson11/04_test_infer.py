import os
import json
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from model import SkeletonTransformer

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_ROOT = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson11"
BATCH_SIZE = 16
TARGET_FRAMES = 30
FRAME_DIM = 132

# 加载标签映射
with open(os.path.join(OUTPUT_ROOT, "label_map.json"), "r", encoding="utf-8") as f:
    label_info = json.load(f)
ID2LABEL = label_info["id2class"]
LABEL_NAMES = list(label_info["class2id"].keys())

# 加载模型
MODEL_PATH = os.path.join(OUTPUT_ROOT, "best_model.pth")
model = SkeletonTransformer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== 工具函数（同预处理） =====================
def extract_skeleton_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    skeleton_seq = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if not results.pose_landmarks:
            frame_feat = np.zeros(FRAME_DIM, dtype=np.float32)
        else:
            frame_feat = []
            for lm in results.pose_landmarks.landmark:
                frame_feat.extend([lm.x, lm.y, lm.z, lm.visibility])
            frame_feat = np.array(frame_feat, dtype=np.float32)
        skeleton_seq.append(frame_feat)
    cap.release()
    return np.array(skeleton_seq)

def resample_sequence(seq, target_len):
    n = len(seq)
    if n == 0:
        return np.zeros((target_len, FRAME_DIM), dtype=np.float32)
    indices = np.linspace(0, n - 1, target_len, dtype=int)
    return seq[indices]

def normalize_skeleton(seq):
    left_hip_idx = 23 * 4
    right_hip_idx = 24 * 4
    left_shoulder_idx = 11 * 4
    right_shoulder_idx = 12 * 4
    for i in range(len(seq)):
        frame = seq[i]
        hip_center_x = (frame[left_hip_idx] + frame[right_hip_idx]) / 2.0
        hip_center_y = (frame[left_hip_idx+1] + frame[right_hip_idx+1]) / 2.0
        shoulder_width = abs(frame[left_shoulder_idx] - frame[right_shoulder_idx])
        if shoulder_width < 1e-6:
            shoulder_width = 1.0
        for j in range(0, FRAME_DIM, 4):
            frame[j] = (frame[j] - hip_center_x) / shoulder_width
            frame[j+1] = (frame[j+1] - hip_center_y) / shoulder_width
    return seq

# ===================== 数据集 =====================
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

# ===================== 测试集评估（混淆矩阵+分类报告） =====================
def evaluate_testset():
    test_data = os.path.join(OUTPUT_ROOT, "X_test.npy")
    test_label = os.path.join(OUTPUT_ROOT, "y_test.npy")
    dataset = SkeletonDataset(test_data, test_label)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_trues = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(labels.cpu().numpy())

    # 计算指标
    test_acc = np.mean(np.array(all_preds) == np.array(all_trues))
    avg_loss = total_loss / len(loader)
    print(f"测试集 Loss: {avg_loss:.4f} | 测试集 Acc: {test_acc:.4f}")

    # 分类报告
    report = classification_report(all_trues, all_preds, target_names=LABEL_NAMES)
    print("\n===== 分类报告 =====")
    print(report)

    # 保存分类报告
    report_path = os.path.join(OUTPUT_ROOT, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predict")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(OUTPUT_ROOT, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"\n📈 混淆矩阵已保存: {cm_path}")
    print(f"📄 分类报告已保存: {report_path}")

# ===================== 单视频推理 =====================
def single_video_infer(video_path):
    print(f"\n===== 开始推理视频: {video_path} =====")
    # 提取+预处理
    raw_seq = extract_skeleton_from_video(video_path)
    fixed_seq = resample_sequence(raw_seq, TARGET_FRAMES)
    norm_seq = normalize_skeleton(fixed_seq)

    # 转为模型输入 [1, 30, 132]
    input_tensor = torch.from_numpy(norm_seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_class = ID2LABEL[str(pred_idx.item())]
    confidence = conf.item()

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2f}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 1. 评估测试集，生成混淆矩阵、分类报告
    evaluate_testset()

    # 2. 单视频推理（替换为你的测试视频路径）
    demo_video = r"D:\迅雷下载\archive\forehand_clear\xxx.mp4"  # 自行修改视频路径
    if os.path.exists(demo_video):
        single_video_infer(demo_video)
    else:
        print("\n提示：请修改代码中 demo_video 为实际视频路径，再执行单样本推理！")