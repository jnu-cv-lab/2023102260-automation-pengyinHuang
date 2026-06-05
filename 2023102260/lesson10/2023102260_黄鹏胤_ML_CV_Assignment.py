# 首行强制指定matplotlib非交互式后端，解决Linux服务端无桌面启动报错
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os

# 固定目标保存路径
save_dir = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson10"
os.makedirs(save_dir, exist_ok=True)

# ======================================
# Task1：Sinusoidal Positional Encoding
# ======================================
def sinusoidal_pos_encoding(max_seq_len: int, embed_dim: int):
    pos_arr = np.arange(max_seq_len)[:, np.newaxis]
    freq_idx = np.arange(0, embed_dim, 2)[np.newaxis, :]
    denominator = np.power(10000, 2 * freq_idx / embed_dim)
    pos_enc = np.zeros((max_seq_len, embed_dim))
    pos_enc[:, 0::2] = np.sin(pos_arr / denominator)
    pos_enc[:, 1::2] = np.cos(pos_arr / denominator)
    return pos_enc

L, D = 50, 64
pe_mat = sinusoidal_pos_encoding(L, D)
plt.figure(figsize=(10, 5), dpi=120)
heat = plt.imshow(pe_mat, cmap="RdBu_r", aspect="auto")
plt.xlabel("Embedding Dimension")
plt.ylabel("Token Sequence Position")
plt.title("Sinusoidal Positional Encoding Heatmap (Task1)")
plt.colorbar(heat)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "task1_sinusoidal_pe.png"))
plt.close()

# ======================================
# Task2：2D Vector Rotation
# ======================================
def rotate_2d_vector(vec: np.ndarray, rad: float):
    rot_matrix = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])
    return rot_matrix @ vec

origin_vec = np.array([1.0, 0.0])
angle_list = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
rot_result = [rotate_2d_vector(origin_vec, ang) for ang in angle_list]

plt.figure(figsize=(6,6), dpi=120)
ax = plt.gca()
ax.set_xlim(-1.3,1.3)
ax.set_ylim(-1.3,1.3)
ax.axhline(y=0, c="black", lw=0.7)
ax.axvline(x=0, c="black", lw=0.7)
color_map = ["red", "orange", "gold", "forestgreen", "royalblue"]
for idx, v in enumerate(rot_result):
    ax.arrow(0,0, v[0],v[1], width=0.022, color=color_map[idx], label=f"θ={angle_list[idx]/np.pi:.2f}π")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("2D Vector Rotation Demo (Task2)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "task2_2d_rotation.png"))
plt.close()

# ======================================
# Task3：High-Dimensional RoPE
# ======================================
def rope_high_dim(x: np.ndarray, pos: int, base=10000):
    D = x.shape[0]
    out = np.zeros_like(x)
    for d_pair in range(0, D, 2):
        half_idx = d_pair // 2
        theta = pos / (base ** (2 * half_idx / D))
        x0, x1 = x[d_pair], x[d_pair+1]
        out[d_pair]   = x0 * np.cos(theta) - x1 * np.sin(theta)
        out[d_pair+1] = x0 * np.sin(theta) + x1 * np.cos(theta)
    return out

np.random.seed(233)
test_dim = 16
embed_vec = np.random.randn(test_dim)
pos_5 = rope_high_dim(embed_vec, pos=5)
pos_9 = rope_high_dim(embed_vec, pos=9)

# ======================================
# Task5：RoPE Relative Position Experiment
# ======================================
def rope_qk_score(q_vec, k_vec, pos_q, pos_k):
    q_rot = rope_high_dim(q_vec, pos_q)
    k_rot = rope_high_dim(k_vec, pos_k)
    return np.dot(q_rot, k_rot)

np.random.seed(42)
D_test = 8
q_base = np.random.randn(D_test)
k_base = np.random.randn(D_test)

rel_offset_1 = 3
s1 = rope_qk_score(q_base, k_base, pos_q=2, pos_k=2+rel_offset_1)
s2 = rope_qk_score(q_base, k_base, pos_q=7, pos_k=7+rel_offset_1)

rel_offset_2 = 5
s3 = rope_qk_score(q_base, k_base, pos_q=1, pos_k=1+rel_offset_2)
s4 = rope_qk_score(q_base, k_base, pos_q=9, pos_k=9+rel_offset_2)

offset_range = list(range(0,13))
score_list = []
fix_q_pos = 4
for off in offset_range:
    s = rope_qk_score(q_base, k_base, fix_q_pos, fix_q_pos + off)
    score_list.append(s)

plt.figure(figsize=(8,4), dpi=120)
plt.plot(offset_range, score_list, marker="o", c="#2255bb", linewidth=1.5)
plt.xlabel("Relative Position Offset (pos_k - pos_q)")
plt.ylabel("Q·K Attention Score after RoPE")
plt.title("RoPE Relative Position Property Verification (Task5)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "task5_relative_pos_verify.png"))
plt.close()

# Print test info
print("===== Task5 RoPE Relative Position Test Result =====")
print(f"Relative Offset={rel_offset_1}, AbsPos1 Score={s1:.4f}, AbsPos2 Score={s2:.4f}")
print(f"Relative Offset={rel_offset_2}, AbsPos1 Score={s3:.4f}, AbsPos2 Score={s4:.4f}")
print("Conclusion: Same relative offset → nearly equal attention score")

# Task6 Conclusion
rope_advantage = """
Q: Why RoPE is more elegant than naive E+pos additive positional encoding?
1. Extrapolation Ability: E+pos uses fixed absolute PE, cannot generalize to sequence longer than training max length;
   RoPE encodes relative position into QK inner-product, naturally extrapolate to unseen long text.
2. Encoding Form: E+pos inject absolute position into embedding statically at input layer;
   RoPE dynamically fuse positional info inside attention via geometric rotation.
3. Inductive Bias: RoPE introduce 2D rotation geometric prior matching sequential relative dependency,
   E+pos is additive with no explicit relative position constraint.
4. Flexibility: RoPE's positional information only affects Q·K calculation, not pollute value(V) embedding;
   E+pos modifies entire token embedding space permanently.
"""
print("\n===== Task6 Analysis =====")
print(rope_advantage)

print(f"\nAll pictures saved at: {save_dir}")