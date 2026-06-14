import os
import json
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ===================== 全局配置 =====================
# 数据集根目录
DATA_ROOT = "/mnt/d/迅雷下载/archive"
# 成果输出路径
OUTPUT_ROOT = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson11"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
# 统一视频帧数
TARGET_FRAMES = 30
# 每帧特征维度（先132，要开396维再改）
FRAME_DIM = 132
# 测试集比例
TEST_RATIO = 0.2
# 类别映射
CLASS_MAP = {
    "forehand_drive": 0,
    "forehand_lift": 1,
    "forehand_net_shot": 2,
    "forehand_clear": 3,
    "backhand_drive": 4,
    "backhand_net_shot": 5
}
ID2LABEL = {v: k for k, v in CLASS_MAP.items()}

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== 工具函数 =====================
def extract_skeleton_from_video(video_path):
    """读取视频，裁剪画面+中心检测+过滤无效骨架，只提取运动员"""
    cap = cv2.VideoCapture(video_path)
    skeleton_seq = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- 关键修复1：裁剪画面，只保留球场中心区域 ----------------
        h, w = frame.shape[:2]
        # 裁掉左右各15%，只保留中间70%的球场区域，过滤裁判/观众
        x1, x2 = int(w * 0.15), int(w * 0.85)
        y1, y2 = int(h * 0.10), int(h * 0.90)
        center_frame = frame[y1:y2, x1:x2]

        # 转为RGB格式
        frame_rgb = cv2.cvtColor(center_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # ---------------- 关键修复2：过滤无效骨架，只保留中心区域的人体 ----------------
        if not results.pose_landmarks:
            # 无检测结果，填充0向量
            frame_feat = np.zeros(132, dtype=np.float32)
        else:
            landmarks = results.pose_landmarks.landmark
            # 用髋部中心判断人体位置，必须在画面中心区域（0.3~0.7范围）
            hip_x = (landmarks[23].x + landmarks[24].x) / 2
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
            if not (0.3 < hip_x < 0.7 and 0.2 < hip_y < 0.8):
                # 骨架不在中心，判定为无效（裁判/观众），填充0向量
                frame_feat = np.zeros(132, dtype=np.float32)
            else:
                # 有效骨架，提取特征
                frame_feat = []
                for lm in landmarks:
                    frame_feat.extend([lm.x, lm.y, lm.z, lm.visibility])
                frame_feat = np.array(frame_feat, dtype=np.float32)

        skeleton_seq.append(frame_feat)
    cap.release()
    return np.array(skeleton_seq, dtype=np.float32)

def resample_sequence(seq, target_len):
    """将不等长序列均匀重采样为固定长度"""
    n = len(seq)
    if n == 0:
        return np.zeros((target_len, FRAME_DIM), dtype=np.float32)
    indices = np.linspace(0, n - 1, target_len, dtype=int)
    return seq[indices]

def normalize_skeleton(seq):
    """骨架归一化：以左右髋部中心为原点，肩宽做尺度归一化"""
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

def add_kinematic_features(seq):
    """速度+加速度特征（后续想冲更高分再开）"""
    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]
    acceleration = np.zeros_like(seq)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    new_seq = np.concatenate([seq, velocity, acceleration], axis=-1)
    return new_seq

# ===================== 主预处理流程 =====================
def main():
    all_sequences = []
    all_labels = []
    for cls_name, cls_id in tqdm(CLASS_MAP.items(), desc="遍历类别"):
        cls_dir = os.path.join(DATA_ROOT, cls_name)
        if not os.path.exists(cls_dir):
            print(f"警告：文件夹 {cls_dir} 不存在，跳过")
            continue
        video_files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        for vid_name in tqdm(video_files, desc=f"处理 {cls_name}", leave=False):
            vid_path = os.path.join(cls_dir, vid_name)
            raw_seq = extract_skeleton_from_video(vid_path)
            fixed_seq = resample_sequence(raw_seq, TARGET_FRAMES)
            norm_seq = normalize_skeleton(fixed_seq)

            # 先关闭动态特征，跑通流程后再开
            # norm_seq = add_kinematic_features(norm_seq)

            all_sequences.append(norm_seq)
            all_labels.append(cls_id)
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    print(f"全部数据形状: X={X.shape}, y={y.shape}")
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=42, stratify=y
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    # 保存到指定成果路径
    np.save(os.path.join(OUTPUT_ROOT, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_ROOT, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_ROOT, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_ROOT, "y_test.npy"), y_test)
    with open(os.path.join(OUTPUT_ROOT, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"class2id": CLASS_MAP, "id2class": ID2LABEL}, f, ensure_ascii=False, indent=2)
    print(f"✅ 数据预处理完成，文件已保存至：{OUTPUT_ROOT}")

if __name__ == "__main__":
    main()