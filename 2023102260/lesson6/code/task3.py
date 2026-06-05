import cv2
import numpy as np

# -------------------------- 1. 配置文件路径 --------------------------
template_path = ".venv-basic/picture/lesson6/box.png"
scene_path = ".venv-basic/picture/lesson6/box_in_scene.png"
output_ransac_result = ".venv-basic/picture/lesson6/RANSAC_match_result.png"

# -------------------------- 2. 读取图像 --------------------------
img1 = cv2.imread(template_path)
img2 = cv2.imread(scene_path)

# -------------------------- 3. ORB 检测 + 匹配（复用任务2代码） --------------------------
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 总匹配数量
num_matches = len(matches)

# -------------------------- 4. 任务3核心：提取对应点坐标 --------------------------
pts1 = []
pts2 = []
for m in matches:
    pts1.append(kp1[m.queryIdx].pt)  # 模板图的点坐标
    pts2.append(kp2[m.trainIdx].pt)  # 场景图的点坐标

# 转成float32格式（findHomography要求）
pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

# -------------------------- 5. 计算单应矩阵 + RANSAC --------------------------
# 要求：cv2.findHomography + cv2.RANSAC + 重投影误差5.0
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=5.0)

# -------------------------- 6. 统计内点 --------------------------
inlier_mask = mask.ravel().tolist()  # 内点掩码
num_inliers = sum(inlier_mask)       # 内点数量
inlier_ratio = num_inliers / num_matches  # 内点比例

# -------------------------- 7. 可视化 RANSAC 筛选后的匹配 --------------------------
img_ransac = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches,
    None,
    matchesMask=inlier_mask,  # 只画内点
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# -------------------------- 8. 保存结果图 --------------------------
cv2.imwrite(output_ransac_result, img_ransac)

# -------------------------- 9. 输出要求的所有信息 --------------------------
print("=" * 60)
print("【任务3 RANSAC 剔除错误匹配 结果】")
print("=" * 60)
print(f"1. 总匹配数量：{num_matches}")
print(f"2. RANSAC内点数量：{num_inliers}")
print(f"3. 内点比例：{inlier_ratio:.2%}")
print("\n4. Homography 矩阵：")
print(np.round(H, 4))  # 保留4位小数，更整洁
print("=" * 60)
print(f"RANSAC筛选后匹配图已保存至：{output_ransac_result}")