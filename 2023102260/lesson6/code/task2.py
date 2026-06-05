import cv2
import numpy as np

# -------------------------- 1. 配置文件路径 --------------------------
# 原图路径 (和上节课一样)
template_path = ".venv-basic/picture/lesson6/box.png"
scene_path = ".venv-basic/picture/lesson6/box_in_scene.png"

# 输出结果保存路径
output_match_result = ".venv-basic/picture/lesson6/ORB_match_result.png"

# -------------------------- 2. 读取图像 --------------------------
img1 = cv2.imread(template_path)
img2 = cv2.imread(scene_path)

# -------------------------- 3. 初始化 ORB 检测器 --------------------------
orb = cv2.ORB_create(nfeatures=1000)

# 检测关键点并计算描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# -------------------------- 4. 创建暴力匹配器 (BFMatcher) --------------------------
# 要求1: cv2.BFMatcher()
# 要求2: ORB 使用 cv2.NORM_HAMMING
# 要求3: crossCheck=True
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# -------------------------- 5. 开始匹配 --------------------------
matches = bf.match(des1, des2)

# -------------------------- 6. 按照匹配距离从小到大排序 --------------------------
# 要求4: 排序
matches = sorted(matches, key = lambda x:x.distance)

# -------------------------- 7. 输出结果 --------------------------
# 要求6: 输出总匹配数量
print("=" * 50)
print(f"【总匹配数量】: {len(matches)} 个")
print("=" * 50)

# -------------------------- 8. 可视化匹配结果 --------------------------
# 要求5: 显示前 30 或 50 个 (这里选前50个, 效果更好)
img_result = cv2.drawMatches(
    img1, kp1,     # 左图(模板)及其特征点
    img2, kp2,     # 右图(场景)及其特征点
    matches[:50],  # 选取前50个最优匹配
    None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 不画未匹配的点
)

# -------------------------- 9. 保存图片 --------------------------
cv2.imwrite(output_match_result, img_result)
print(f"匹配结果图已保存至: {output_match_result}")

# 显示图片 (可选, 方便查看)
cv2.imshow('ORB Feature Matching Result', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()