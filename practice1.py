import cv2
import numpy as np
import math

def calculate_psnr(img_original, img_reconstructed):
    """
    计算两张图像的峰值信噪比（PSNR）
    :param img_original: 原始图像
    :param img_reconstructed: 重建后的图像
    :return: PSNR 值（dB），值越大表示失真越小
    """
    # 确保两张图像尺寸一致
    if img_original.shape != img_reconstructed.shape:
        raise ValueError(" 原始图像和重建图像尺寸不一致，无法计算 PSNR！")
    
    # 计算均方误差（MSE）
    mse = np.mean((img_original.astype(np.float32) - img_reconstructed.astype(np.float32)) ** 2)
    
    # 如果 MSE 为 0，说明图像完全相同，PSNR 无穷大（返回 100 作为标识）
    if mse == 0:
        return 100.0
    
    # 计算 PSNR（像素值范围 0-255）
    max_pixel = 255.0
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_value


img = cv2.imread("/home/hpy3378092/cv-course/.venv-basic/picture/c罗和弗格森.jpg")
img_ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
Y,Cr,Cb = cv2.split(img_ycrcb)
# 打印Y/Cb/Cr的通道的核心数值
print("=== Y/Cb/Cr 通道数值 ===")
print(f"Y通道：尺寸{Y.shape} | 像素范围[{np.min(Y)}, {np.max(Y)}] | 均值{np.mean(Y):.2f}")
print(f"Cr通道：尺寸{Cr.shape} | 像素范围[{np.min(Cr)}, {np.max(Cr)}] | 均值{np.mean(Cr):.2f}")
print(f"Cb通道：尺寸{Cb.shape} | 像素范围[{np.min(Cb)}, {np.max(Cb)}] | 均值{np.mean(Cb):.2f}")

h,w = Cb.shape
Cb_down = cv2.resize(Cb, (w//2, h//2), interpolation=cv2.INTER_AREA)
Cr_down = cv2.resize(Cr, (w//2, h//2), interpolation=cv2.INTER_AREA)
print(" Cb/Cr 下采样后形状:", Cb_down.shape)

# ---------------------- 4. 插值方法恢复原尺寸 ----------------------
# 最近邻插值恢复（也可尝试 INTER_CUBIC/INTER_NEAREST 对比效果）
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=cv2.INTER_NEAREST)
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=cv2.INTER_NEAREST)

# ---------------------- 5. 与原 Y 通道重建图像 ----------------------
img_ycrcb_recon = cv2.merge([Y, Cr_up, Cb_up])  # 合并 Y + 恢复后的 Cr/Cb
img_recon = cv2.cvtColor(img_ycrcb_recon, cv2.COLOR_YCrCb2BGR)  # 转回 BG

psnr_result = calculate_psnr(img, img_recon)
print(f" 原始图像 vs 重建图像 PSNR 值: {psnr_result:.2f} dB")

cv2.imwrite(".venv-basic/picture" + "reconstructed2.jpg", img_recon)

# 根据打印结果可知，
# 下采样的影响：
# 对 Cr/Cb 色度通道做 2 倍下采样，能减少 75% 的色度数据量（节省存储 / 传输成本）；
# 人眼对亮度（Y 通道）敏感、对色度（Cr/Cb）不敏感，所以下采样后肉眼几乎看不出画质损失。
# 插值恢复的影响：
# 用的是「最近邻插值」，这种方法速度最快，但会轻微失真（比如边缘出现锯齿、少量色块）；
# 从 PSNR 数值来看：如果 PSNR=51.91dB＞30dB，人眼基本分不清原始图和重建图。
