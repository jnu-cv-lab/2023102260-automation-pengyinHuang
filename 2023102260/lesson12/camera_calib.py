import cv2
import numpy as np
import glob
import os

# ====================== 1. 全局参数 ======================
# 棋盘格内角点规格 9×6
CHESSBOARD_SIZE = (9, 6)
# 屏幕棋盘单格物理边长(mm)，自行修改为你实际测量值
SQUARE_SIZE_MM = 26
# 原始未处理图片存放路径
IMG_SRC_ROOT = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson12/picture_no_process"
# 所有处理后图片输出目录
IMG_OUT_ROOT = "/home/hpy3378092/cv-course/.venv-basic/picture/lesson12/final_picture"
# 匹配jpg/png图片
IMG_PATTERN = os.path.join(IMG_SRC_ROOT, "*.[jp][pn]g")
# 亚像素角点迭代终止条件
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 适配屏幕棋盘的角点检测标志
FIND_FLAG = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FILTER_QUADS
             | cv2.CALIB_CB_FAST_CHECK)

# 自动创建输出文件夹
if not os.path.exists(IMG_OUT_ROOT):
    os.makedirs(IMG_OUT_ROOT)
    print(f"已创建输出文件夹：{IMG_OUT_ROOT}")

# ====================== 2. 生成棋盘三维世界坐标 ======================
objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

# 存储有效角点数据
obj_points = []
img_points = []
valid_img_paths = []

# ====================== 3. 批量读取图片、检测角点 ======================
image_paths = glob.glob(IMG_PATTERN)
print(f"共找到 {len(image_paths)} 张待标定图片")

for idx, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    if img is None:
        print(f"跳过无效图片：{img_path}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测棋盘内角点
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags=FIND_FLAG)

    if ret:
        # 亚像素精度优化
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        obj_points.append(objp)
        img_points.append(corners_sub)
        valid_img_paths.append(img_path)

        # 绘制并保存角点图到输出目录
        draw_img = img.copy()
        cv2.drawChessboardCorners(draw_img, CHESSBOARD_SIZE, corners_sub, ret)
        save_draw_path = os.path.join(IMG_OUT_ROOT, f"corner_draw_{idx}.jpg")
        cv2.imwrite(save_draw_path, draw_img)
        print(f"✅ 检测成功：{os.path.basename(img_path)}，角点图已保存")
    else:
        print(f"❌ 未检测到完整棋盘角点：{os.path.basename(img_path)}")

# 移除15张强制校验，仅做提示
valid_num = len(obj_points)
print(f"\n当前可用于标定的有效图片数量：{valid_num} 张")
if valid_num < 15:
    print("提示：有效图片不足15张，标定精度会下降，仅用于作业调试")

# 无有效图片直接退出
if valid_num == 0:
    raise Exception("没有任何一张图片成功检测棋盘角点，无法执行标定！")

# ====================== 4. 执行相机标定 ======================
# 取最后一张图获取图像尺寸
h_tmp, w_tmp = cv2.imread(valid_img_paths[-1]).shape[:2]
rms_error, K, D, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, (w_tmp, h_tmp), None, None
)

# ====================== 5. 计算平均重投影误差 ======================
total_err = 0.0
for i in range(len(obj_points)):
    proj_pts, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, D)
    err = cv2.norm(img_points[i], proj_pts, cv2.NORM_L2) / len(proj_pts)
    total_err += err
avg_reproj_err = total_err / len(obj_points)

# ====================== 6. 终端输出标定核心结果（报告直接复制） ======================
print("\n==================== 相机标定结果 ====================")
print(f"1. RMS整体重投影误差: {rms_error:.4f} px")
print(f"2. 单张图像平均重投影误差: {avg_reproj_err:.4f} px")
print("\n3. 相机内参矩阵 K：")
print(K)
print("\n4. 畸变系数 D = [k1, k2, p1, p2, k3]：")
print(D.flatten())

# 保存全部标定参数到输出文件夹
npz_save_path = os.path.join(IMG_OUT_ROOT, "calib_params.npz")
np.savez(npz_save_path, K=K, D=D, rvecs=rvecs, tvecs=tvecs)
print(f"\n标定参数文件已保存至：{npz_save_path}")

# ====================== 7. 生成原图+去畸变对比图 ======================
demo_img_path = valid_img_paths[0]
demo_img = cv2.imread(demo_img_path)
h, w = demo_img.shape[:2]

# 优化内参矩阵，适配去畸变
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
# 图像去畸变
undist_img = cv2.undistort(demo_img, K, D, None, new_K)
# 裁剪画面黑边
x, y, w_roi, h_roi = roi
undist_crop = undist_img[y:y+h_roi, x:x+w_roi]

# 保存三张对比图片到输出目录
cv2.imwrite(os.path.join(IMG_OUT_ROOT, "demo_original.jpg"), demo_img)
cv2.imwrite(os.path.join(IMG_OUT_ROOT, "demo_undistort_full.jpg"), undist_img)
cv2.imwrite(os.path.join(IMG_OUT_ROOT, "demo_undistort_crop.jpg"), undist_crop)

print("\n去畸变对比图保存完成，路径：")
print(f"原图：{os.path.join(IMG_OUT_ROOT, 'demo_original.jpg')}")
print(f"完整去畸变图（含黑边）：{os.path.join(IMG_OUT_ROOT, 'demo_undistort_full.jpg')}")
print(f"裁剪黑边去畸变图：{os.path.join(IMG_OUT_ROOT, 'demo_undistort_crop.jpg')}")

print(f"\n标定流程全部执行完毕！所有结果图片、参数文件存放目录：{IMG_OUT_ROOT}")