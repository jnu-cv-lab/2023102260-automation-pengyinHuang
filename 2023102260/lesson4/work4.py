import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 完全去掉中文，永无报错 --------------------------

# -------------------------- 工具函数 --------------------------
def compute_gradient_rms(img_block):
    grad_x = cv2.Scharr(img_block, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(img_block, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    E_grad_sq = np.mean(grad_mag**2)
    var_I = np.var(img_block)
    if var_I < 1e-6:
        return 0.0
    f_rms_sq = E_grad_sq / (4 * np.pi**2 * var_I)
    return np.sqrt(f_rms_sq)

def compute_fft_95energy_freq(img_block):
    H, W = img_block.shape
    f = np.fft.fft2(img_block)
    f_shift = np.fft.fftshift(f)
    power_spectrum = np.abs(f_shift)**2

    u = np.fft.fftfreq(W, d=1)
    v = np.fft.fftfreq(H, d=1)
    u_shift = np.fft.fftshift(u)
    v_shift = np.fft.fftshift(v)
    u_grid, v_grid = np.meshgrid(u_shift, v_shift)
    freq_mag = np.sqrt(u_grid**2 + v_grid**2)

    power_flat = power_spectrum.flatten()
    freq_flat = freq_mag.flatten()

    sort_idx = np.argsort(freq_flat)
    power_sorted = power_flat[sort_idx]
    freq_sorted = freq_flat[sort_idx]

    total_energy = np.sum(power_sorted)
    cumulative_energy = np.cumsum(power_sorted)
    idx_95 = np.argmax(cumulative_energy >= 0.95 * total_energy)
    return freq_sorted[idx_95]

def split_image_blocks(img, block_size=32):
    H, W = img.shape
    blocks = []
    for i in range(0, H - block_size + 1, block_size):
        for j in range(0, W - block_size + 1, block_size):
            block = img[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    img_path = ".venv-basic/picture/【哲风壁纸】7号球衣-Cristiano.png"
    save_dir = "/home/hpy3378092/cv-course/.venv-basic/picture/"

    # 读取图片
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # 分块
    block_size = 32
    blocks = split_image_blocks(img, block_size=block_size)
    print(f"Image split into {len(blocks)} blocks of {block_size}x{block_size}")

    # 计算频率
    grad_f_list = []
    fft_f_list = []
    for idx, block in enumerate(blocks):
        f_grad = compute_gradient_rms(block)
        f_fft = compute_fft_95energy_freq(block)
        grad_f_list.append(f_grad)
        fft_f_list.append(f_fft)

        if idx < 10:
            print(f"Block {idx+1:2d} | Grad f: {f_grad:6.4f} | FFT f: {f_fft:6.4f}")

    # 统计
    grad_f = np.array(grad_f_list)
    fft_f = np.array(fft_f_list)
    corr = np.corrcoef(grad_f, fft_f)[0, 1]
    mae = np.mean(np.abs(grad_f - fft_f))

    print("\n===== Result =====")
    print(f"Correlation: {corr:.4f}")
    print(f"MAE: {mae:.4f}")

    # 绘图 1：对比图
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.scatter(fft_f, grad_f, s=8, alpha=0.7)
    plt.plot([0, np.max(fft_f)], [0, np.max(fft_f)], 'r--', lw=1)
    plt.xlabel("FFT 95% Energy Freq")
    plt.ylabel("Gradient Method Freq")
    plt.title(f"Correlation = {corr:.4f}")
    plt.grid(alpha=0.3)

    plt.subplot(122)
    plt.hist(grad_f - fft_f, bins=30, alpha=0.7, edgecolor="k")
    plt.xlabel("Error (Grad - FFT)")
    plt.ylabel("Count")
    plt.title(f"Error Distribution | MAE = {mae:.4f}")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir + "freq_comparison.png", dpi=300)
    plt.close()

    # 绘图 2：块 + 功率谱
    sample_block = blocks[0]
    f_grad_s = compute_gradient_rms(sample_block)
    f_fft_s = compute_fft_95energy_freq(sample_block)

    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.imshow(sample_block, cmap='gray')
    plt.title(f"Sample Block | Grad: {f_grad_s:.4f} | FFT: {f_fft_s:.4f}")
    plt.axis('off')

    f = np.fft.fft2(sample_block)
    f_shift = np.fft.fftshift(f)
    power = 20 * np.log(np.abs(f_shift) + 1)
    plt.subplot(122)
    plt.imshow(power, cmap='gray')
    plt.title("Power Spectrum")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_dir + "sample_block_result.png", dpi=300)
    plt.close()

    print("\n✅ All done! Images saved to:\n" + save_dir)