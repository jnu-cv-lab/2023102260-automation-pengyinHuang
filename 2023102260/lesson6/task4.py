import cv2
import numpy as np
import os

# -------------------------- 1. 修正路径：使用绝对路径，确保读取成功 --------------------------
# 用你的实际路径，确保和你的文件位置一致
# 这里我用你终端显示的路径
base_dir = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson6"
template_path = os.path.join(base_dir, "box.png")
scene_path = os.path.join(base_dir, "box_in_scene.png")

# 输出文件路径
output_task2 = os.path.join(base_dir, "task2_match.png")
output_task3 = os.path.join(base_dir, "task3_ransac.png")
output_task4 = os.path.join(base_dir, "task4_final.png")

# -------------------------- 2. 读取图像并检查 --------------------------
img1 = cv2.imread(template_path)
img2 = cv2.imread(scene_path)

# 关键：读取失败直接退出，不执行后面的代码
if img1 is None:
    print(f"❌ 错误：无法读取 box.png，路径：{template_path}")
    print("请检查：1. 文件名是否大小写正确 2. 文件是否存在 3. 路径是否正确")
    exit()
if img2 is None:
    print(f"❌ 错误：无法读取 box_in_scene.png，路径：{scene_path}")
    exit()

print(f"✅ 图像读取成功！")
print(f"box.png 尺寸: {img1.shape}")
print(f"box_in_scene.png 尺寸: {img2.shape}")

# -------------------------- 任务1：ORB特征检测 --------------------------
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print(f"\n【任务1结果】")
print(f"box.png 关键点数量: {len(kp1)}, 描述子维度: {des1.shape}")
print(f"box_in_scene.png 关键点数量: {len(kp2)}, 描述子维度: {des2.shape}")

# -------------------------- 任务2：ORB特征匹配 --------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

print(f"\n【任务2结果】")
print(f"总匹配数量: {len(matches)}")

# 保存前50个匹配图
img_match = cv2.drawMatches(
    img1, kp1, img2, kp2, matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(output_task2, img_match)
print(f"✅ 任务2匹配图已保存: {output_task2}")

# -------------------------- 任务3：RANSAC剔除错误匹配 --------------------------
# 提取匹配点坐标
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 计算Homography矩阵
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
inliers_mask = mask.ravel().tolist()
num_inliers = sum(inliers_mask)
inlier_ratio = num_inliers / len(matches)

print(f"\n【任务3结果】")
print(f"总匹配数量: {len(matches)}")
print(f"RANSAC内点数量: {num_inliers}")
print(f"内点比例: {inlier_ratio:.2%}")
print(f"Homography矩阵:\n{np.round(H, 4)}")

# 保存RANSAC匹配图
img_ransac = cv2.drawMatches(
    img1, kp1, img2, kp2, matches, None,
    matchesMask=inliers_mask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(output_task3, img_ransac)
print(f"✅ 任务3 RANSAC匹配图已保存: {output_task3}")

# -------------------------- 任务4：目标定位 --------------------------
h, w = img1.shape[:2]
# 获取box.png的四个角点
pts_corner = np.float32([
    [0, 0],          # 左上
    [w, 0],          # 右上
    [w, h],          # 右下
    [0, h]           # 左下
]).reshape(-1, 1, 2)

# 投影到场景图
pts_projected = cv2.perspectiveTransform(pts_corner, H)

# 在场景图上画框
img_final = img2.copy()
cv2.polylines(
    img_final,
    [np.int32(pts_projected)],
    True,          # 闭合
    (0, 255, 0),   # 绿色
    5,             # 线宽
    cv2.LINE_AA
)
cv2.imwrite(output_task4, img_final)

print(f"\n【任务4结果】")
print("✅ 目标定位图已保存，绿色边框应精准包围场景中的box！")
print(f"文件路径: {output_task4}")
print("\n所有任务完成！请检查生成的三张图片！")