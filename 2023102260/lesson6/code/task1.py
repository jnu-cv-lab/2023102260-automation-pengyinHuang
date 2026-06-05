import cv2
import numpy as np

# -------------------------- 1. 图片路径配置 --------------------------
# 原图路径
template_path = ".venv-basic/picture/lesson6/box.png"          # 模板图
scene_path = ".venv-basic/picture/lesson6/box_in_scene.png"    # 场景图

# 输出保存路径
output_template = ".venv-basic/picture/lesson6/box_keypoints.png"
output_scene = ".venv-basic/picture/lesson6/box_in_scene_keypoints.png"

# -------------------------- 2. 创建 ORB 检测器 --------------------------
# nfeatures=1000 满足要求
orb = cv2.ORB_create(nfeatures=1000)

# -------------------------- 3. 读取图像 --------------------------
img_template = cv2.imread(template_path)
img_scene = cv2.imread(scene_path)

# 检查图片是否读取成功
if img_template is None:
    print("错误：无法读取 box.png，请检查路径")
if img_scene is None:
    print("错误：无法读取 box_in_scene.png，请检查路径")

# -------------------------- 4. 检测关键点 + 计算描述子 --------------------------
# detectAndCompute 一步到位
kp1, des1 = orb.detectAndCompute(img_template, None)
kp2, des2 = orb.detectAndCompute(img_scene, None)

# -------------------------- 5. 可视化关键点 --------------------------
# flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 画出带方向、大小的特征点
img_template_kp = cv2.drawKeypoints(img_template, kp1, None, 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_scene_kp = cv2.drawKeypoints(img_scene, kp2, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# -------------------------- 6. 保存图片 --------------------------
cv2.imwrite(output_template, img_template_kp)
cv2.imwrite(output_scene, img_scene_kp)

# -------------------------- 7. 打印要求输出的信息 --------------------------
print("="*50)
print("【box.png 结果】")
print(f"关键点数量：{len(kp1)}")
print(f"描述子维度：{des1.shape}")
print("-"*50)
print("【box_in_scene.png 结果】")
print(f"关键点数量：{len(kp2)}")
print(f"描述子维度：{des2.shape}")
print("="*50)
print(f"可视化图片已保存至：\n{output_template}\n{output_scene}")