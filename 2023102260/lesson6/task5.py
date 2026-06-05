import cv2
import numpy as np
import os

# ====================== 路径（完全按你的要求）======================
base = ".venv-basic/picture/lesson6"
img1_path = os.path.join(base, "box.png")
img2_path = os.path.join(base, "box_in_scene.png")

# 读取图像
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 要测试的三组参数
nfeatures_list = [500, 1000, 2000]

# 保存最终对比结果
result_table = []

print("==================== 任务6：ORB 参数对比实验 ====================\n")

for nfeat in nfeatures_list:
    print(f"\n========== 正在测试 nfeatures = {nfeat} ==========")

    # 1. 创建ORB
    orb = cv2.ORB_create(nfeatures=nfeat)

    # 2. 检测特征
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    kp_num1 = len(kp1)
    kp_num2 = len(kp2)

    # 3. 匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_num = len(matches)

    # 4. RANSAC
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inlier_num = np.sum(mask)
    inlier_ratio = inlier_num / match_num

    # 5. 判断是否定位成功（内点比例 > 0.5 即算成功）
    success = "✅ 成功" if inlier_ratio > 0.5 else "❌ 失败"

    # 输出
    print(f"模板关键点：{kp_num1}")
    print(f"场景关键点：{kp_num2}")
    print(f"总匹配数：{match_num}")
    print(f"RANSAC内点：{inlier_num}")
    print(f"内点比例：{inlier_ratio:.2%}")
    print(f"定位结果：{success}")

    # 存入表格
    result_table.append([
        nfeat, kp_num1, kp_num2, match_num, inlier_num, round(inlier_ratio,4), success
    ])

    # ==================== 保存定位图 ====================
    h, w = img1.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    try:
        proj = cv2.perspectiveTransform(corners, H)
        res = img2.copy()
        cv2.polylines(res, [np.int32(proj)], True, (0,255,0), 4)
        cv2.imwrite(os.path.join(base, f"task6_loc_{nfeat}.png"), res)
    except:
        pass

# ===================== 最终实验表格 =====================
print("\n\n")
print("="*80)
print("✅ 任务6 最终实验结果表")
print("="*80)
print(f"{'nfeatures':<10}{'模板点数':<10}{'场景点数':<10}{'匹配数':<10}{'内点数':<10}{'内点比例':<10}{'定位成功'}")
for row in result_table:
    print(f"{row[0]:<10}{row[1]:<10}{row[2]:<10}{row[3]:<10}{row[4]:<10}{row[5]:<10.2%}{row[6]}")
print("="*80)