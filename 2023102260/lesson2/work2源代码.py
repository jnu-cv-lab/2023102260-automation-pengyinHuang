import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log10, sqrt

# ===================== 路径配置 =====================
SAVE_DIR = ".venv-basic/picture"
IMAGE_PATHS = [
    ".venv-basic/picture/25皇马.jpg",
    ".venv-basic/picture/灰度1.png",
    ".venv-basic/picture/灰度2.png"
]
os.makedirs(SAVE_DIR, exist_ok=True)

# 滤波参数
FILTER_SIZES = [3, 5, 7]
GAUSSIAN_SIGMAS = [0, 1, 2]

# ===================== 评价指标 =====================
def calculate_psnr(original, processed):
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    if mse < 1e-6:
        return 100.0
    return round(20 * log10(255.0 / sqrt(mse)), 2)

def calculate_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = hist / (hist.sum() + 1e-7)
    ent = -np.sum(hist * np.log2(hist + 1e-7))
    return round(ent, 2)

# ===================== 绘图+保存 =====================
def save_figure(orig_img, res_dict, metrics, base_name):
    plt.switch_backend("Agg")  # 无图形界面可用
    keys = list(res_dict.keys())
    for idx, name in enumerate(keys):
        img = res_dict[name]
        psnr = metrics[name]["PSNR"]
        ent = metrics[name]["熵"]

        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"{name}\nPSNR:{psnr} | Entropy:{ent}", fontsize=12)
        axes[0].axis("off")

        axes[1].hist(img.ravel(), 256, [0,256], color="gray")
        axes[1].set_title("Histogram")
        axes[1].set_xlim(0,256)

        save_path = os.path.join(SAVE_DIR, f"{base_name}_{idx}_{name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"已保存：{save_path}")

# ===================== 图像处理主逻辑 =====================
def process_one(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"读取失败：{img_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    res = {"原图": gray.copy()}

    # 1.全局直方图均衡
    res["全局直方图均衡"] = cv2.equalizeHist(gray)

    # 2.均值滤波多尺度
    for k in FILTER_SIZES:
        res[f"均值滤波{k}x{k}"] = cv2.blur(gray, (k,k))

    # 3.高斯滤波多参数
    for k in [3,5]:
        for s in GAUSSIAN_SIGMAS:
            res[f"高斯{k}x{k}_sig{s}"] = cv2.GaussianBlur(gray, (k,k), s)

    # 4.组合：滤波→均衡
    m5 = cv2.blur(gray,(5,5))
    res["均值5x5→均衡"] = cv2.equalizeHist(m5)
    g5 = cv2.GaussianBlur(gray,(5,5),1)
    res["高斯5x5→均衡"] = cv2.equalizeHist(g5)

    # 5.组合：均衡→滤波
    eq = cv2.equalizeHist(gray)
    res["均衡→均值5x5"] = cv2.blur(eq,(5,5))
    res["均衡→高斯5x5"] = cv2.GaussianBlur(eq,(5,5),1)

    # 计算指标
    metrics = {}
    for k,v in res.items():
        metrics[k] = {
            "PSNR": calculate_psnr(gray, v),
            "熵": calculate_entropy(v)
        }

    # 保存所有图+直方图
    save_figure(gray, res, metrics, base_name)

# ===================== 批量执行 =====================
if __name__ == "__main__":
    for p in IMAGE_PATHS:
        process_one(p)
    print("✅ 全部处理完成，结果已存入 .venv-basic/picture")
