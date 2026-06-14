import os
import cv2
import mediapipe as mp
import numpy as np
import torch

# ==================== 全局配置（务必修改为你自己的路径和维度）====================
# 待测试视频绝对路径
VIDEO_PATH = "/mnt/d/迅雷下载/archive/forehand_net_shot/003.mp4"
# 输出文件路径
SAVE_VIDEO_PATH = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson11/skeleton_cls_demo.mp4"
SAVE_FRAME_PATH = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson11/skeleton_cls_frame.jpg"
# 模型路径
MODEL_PATH = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson11/best_model.pth"
# 预处理参数
TARGET_FRAMES = 30
FRAME_DIM = 132   # 132 / 396 二选一，和项目保持一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 类别映射
CLASS_MAP = {
    0: "forehand_drive",
    1: "forehand_lift",
    2: "forehand_net_shot",
    3: "forehand_clear",
    4: "backhand_drive",
    5: "backhand_net_shot"
}
# =======================================================================

# 导入模型
import sys
sys.path.append(os.path.dirname(__file__))
from model import SkeletonTransformer

# 全局初始化 Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
draw_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
conn_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)

# 全局姿态实例（统一使用，解决变量未定义问题）
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 预处理函数
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
            frame_feat = np.zeros(132, dtype=np.float32)
        else:
            landmarks = results.pose_landmarks.landmark
            frame_feat = []
            for lm in landmarks:
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
        for j in range(0, 132, 4):
            frame[j] = (frame[j] - hip_center_x) / shoulder_width
            frame[j+1] = (frame[j+1] - hip_center_y) / shoulder_width
    return seq

def add_kinematic_features(seq):
    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]
    acceleration = np.zeros_like(seq)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    new_seq = np.concatenate([seq, velocity, acceleration], axis=-1)
    return new_seq

# 模型推理
def predict_action(seq_data):
    model = SkeletonTransformer().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    input_tensor = torch.from_numpy(seq_data).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        _, pred_idx = torch.max(logits, dim=1)
    return CLASS_MAP[pred_idx.item()]

def main():
    # 1. 特征提取 + 推理动作类别
    raw_seq = extract_skeleton_from_video(VIDEO_PATH)
    fixed_seq = resample_sequence(raw_seq, TARGET_FRAMES)
    norm_seq = normalize_skeleton(fixed_seq)

    # 396维特征开启下面这行，132维请注释掉
    # norm_seq = add_kinematic_features(norm_seq)

    pred_label = predict_action(norm_seq)
    print(f"🤖 模型预测动作: {pred_label}")

    # 2. 绘制骨架、标注类别并保存视频/截图
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(SAVE_VIDEO_PATH, fourcc, fps, (w, h))

    frame_count = 0
    save_flag = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # 绘制骨架
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                draw_spec, conn_spec
            )
        # 画面标注预测结果
        cv2.putText(frame, f"Action: {pred_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 保存单帧截图
        if frame_count == 10 and not save_flag:
            cv2.imwrite(SAVE_FRAME_PATH, frame)
            save_flag = True

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ 带分类的骨架视频已保存: {SAVE_VIDEO_PATH}")
    print(f"✅ 带分类的骨架截图已保存: {SAVE_FRAME_PATH}")

if __name__ == "__main__":
    main()