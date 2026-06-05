import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, dct
import os

# ===================== 路径配置 =====================
SAVE_DIR = "/home/hpy3378092/cv-course/.venv-basic/picture"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== Linux 中文终极解决 =====================
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 原始信号 =====================
x = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
N = len(x)
n = np.arange(N)

# ===================== 1. 绘制 原始信号 =====================
plt.figure(figsize=(8, 4))
plt.stem(n, x, linefmt='b-', markerfmt='ro', basefmt='k-')
plt.title("Original Signal", fontsize=14)
plt.xlabel("n")
plt.ylabel("x(n)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "original_signal.png"), dpi=300)
plt.close()
print("✅ 原始信号已保存")

# ===================== 2. 延拓对比 =====================
x_periodic = np.tile(x, 3)
n_periodic = np.arange(len(x_periodic))

x_symmetric = np.concatenate([x, x[-2::-1]])
x_symmetric_full = np.tile(x_symmetric, 3)
n_symmetric = np.arange(len(x_symmetric_full))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.stem(n_periodic, x_periodic, basefmt=' ', linefmt='b-', markerfmt='bo', label='Periodic extension')
ax1.axvline(N-0.5, c='r', ls='--', label='Boundary')
ax1.set_title('DFT Periodic Extension', fontsize=14)
ax1.set_xlabel('n')
ax1.set_ylabel('Amplitude')
ax1.grid(alpha=0.3)
ax1.legend()

ax2.stem(n_symmetric, x_symmetric_full, basefmt=' ', linefmt='g-', markerfmt='go', label='Even symmetric extension')
ax2.axvline(N-0.5, c='r', ls='--', label='Boundary')
ax2.set_title('DCT Even Symmetric Extension', fontsize=14)
ax2.set_xlabel('n')
ax2.set_ylabel('Amplitude')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "extension_compare.png"), dpi=300)
plt.close()
print("✅ 延拓对比图已保存")

# ===================== 3. 频谱与能量集中性 =====================
X_dft = fft(x)
X_dft_amp = np.abs(X_dft)
energy_dft = np.sum(X_dft_amp**2)
dft_energy_ratio = np.cumsum(X_dft_amp**2) / energy_dft

X_dct = dct(x, type=2, norm='ortho')
X_dct_amp = np.abs(X_dct)
energy_dct = np.sum(X_dct_amp**2)
dct_energy_ratio = np.cumsum(X_dct_amp**2) / energy_dct

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

ax1.stem(range(N), X_dft_amp, basefmt=' ', linefmt='b-', markerfmt='bo')
ax1.set_title('DFT Amplitude Spectrum', fontsize=14)
ax1.set_xlabel('k')
ax1.set_ylabel('|X(k)|')
ax1.grid(alpha=0.3)

ax2.stem(range(N), X_dct_amp, basefmt=' ', linefmt='g-', markerfmt='go')
ax2.set_title('DCT Amplitude Spectrum', fontsize=14)
ax2.set_xlabel('u')
ax2.set_ylabel('|F(u)|')
ax2.grid(alpha=0.3)

ax3.plot(range(1, N+1), dft_energy_ratio, 'b-o', label='DFT')
ax3.plot(range(1, N+1), dct_energy_ratio, 'g-s', label='DCT')
ax3.set_title('Energy Concentration', fontsize=14)
ax3.set_xlabel('k')
ax3.set_ylabel('Energy Ratio')
ax3.legend()
ax3.grid(alpha=0.3)

ax4.stem(range(N), X_dft_amp**2/energy_dft, basefmt=' ', linefmt='b-', markerfmt='bo', label='DFT')
ax4.stem(range(N), X_dct_amp**2/energy_dct, basefmt=' ', linefmt='g--', markerfmt='gs', label='DCT')
ax4.set_title('Normalized Energy Distribution', fontsize=14)
ax4.set_xlabel('Frequency')
ax4.set_ylabel('Energy')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "spectrum_compare.png"), dpi=300)
plt.close()
print("✅ 频谱能量图已保存")

print("\n🎉 全部完成！原始信号 + 延拓 + 频谱 全部生成！")