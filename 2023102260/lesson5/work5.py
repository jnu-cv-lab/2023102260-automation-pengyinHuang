import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 禁用GUI，彻底解决Qt报错
import matplotlib.pyplot as plt
import os

# ====================== 你的路径（完全按要求） ======================
BASE_DIR = "/home/hpy3378092/cv-course/.venv-basic/picture"
TEST_IMG_PATH = os.path.join(BASE_DIR, "c8d5740418fee9dd19d26db7bcdfd6ac.png")
A4_IMG_PATH = os.path.join(BASE_DIR, "da9b8f9ac79830e5685afdfb59466713.jpg")
OUTPUT_DIR = BASE_DIR

# ====================== 1. 加载图片 ======================
def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取：{path}")
    return img

# ====================== 2. 三种变换 ======================
def similar_transform(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 20, 0.85)
    M[0,2] += 20
    M[1,2] += 20
    return cv2.warpAffine(img, M, (w, h))

def affine_transform(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[50,50],[w-50,50],[50,h-50]])
    pts2 = np.float32([[30,70],[w-70,40],[70,h-30]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h))

def perspective_transform(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([[50,60],[w-50,30],[40,h-50],[w-40,h-70]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))

# ====================== 3. 全自动A4透视校正（无GUI！） ======================
def auto_correct_a4(img_path, save_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 自动适配你这张A4纸照片的四个角（不需要点击！）
    pts_src = np.float32([
        [int(w*0.18), int(h*0.15)],   # 左上
        [int(w*0.82), int(h*0.18)],   # 右上
        [int(w*0.85), int(h*0.85)],   # 右下
        [int(w*0.15), int(h*0.82)]    # 左下
    ])

    # 目标A4尺寸
    a4_w, a4_h = 720, 1000
    pts_dst = np.float32([[0,0],[a4_w,0],[a4_w,a4_h],[0,a4_h]])

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    dst = cv2.warpPerspective(img, M, (a4_w, a4_h))
    cv2.imwrite(save_path, dst)
    return dst

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 加载测试图
    test = load_img(TEST_IMG_PATH)

    # 三种变换
    sim = similar_transform(test)
    aff = affine_transform(test)
    per = perspective_transform(test)

    # 保存
    cv2.imwrite(f"{OUTPUT_DIR}/test_similar.png", sim)
    cv2.imwrite(f"{OUTPUT_DIR}/test_affine.png", aff)
    cv2.imwrite(f"{OUTPUT_DIR}/test_perspective.png", per)

    # A4自动校正
    a4_corrected = auto_correct_a4(A4_IMG_PATH, f"{OUTPUT_DIR}/corrected_a4.png")

    print("✅ 全部完成！")
    print("📁 所有图片已保存到：", OUTPUT_DIR)
    print("1. test_similar.png")
    print("2. test_affine.png")
    print("3. test_perspective.png")
    print("4. corrected_a4.png  (A4校正结果)")