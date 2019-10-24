import cv2
import numpy as np
import os

# 使用matlab得到的内参
# mtx = np.array([[509.0867, 0, 311.2543],
#                 [0, 511.3800, 228.1676],
#                 [0, 0, 1]])
# 使用matlab得到的相机畸变矩阵
# dist = np.array([[0.1588, -0.1978, 0, 0, 0]])

# 读取相机内参和畸变矩阵
with np.load('camera_parameters/intrinsic_parameters.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

# 棋盘底部与搭载相机的物体的距离（x轴方向距离，且单位为mm）
checkerboard_robot_length = 2000.
# 每个网格的长度（单位毫米），它只会影响平移矩阵的大小
grid_length = 25.
# 棋盘在相机的镜头中必须是横放的，且x轴将棋盘分为对称的两份
# 棋盘长边上的角点个数
x_nums = 8
# 棋盘短边上的角点个数
y_nums = 6

# 标定图像
def calibration_photo(image):
    # 设置(生成)标定图在世界坐标中的坐标
    world_point = np.zeros((x_nums * y_nums, 3), np.float32)  # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素

    # 棋盘左上角的点相对与设定的世界坐标系的原点的位置
    x_place = checkerboard_robot_length + y_nums * grid_length
    y_place = (x_nums -  1) * grid_length / 2
    start_place = np.array([x_place, y_place, 0])
    world_point[:, 1] = (np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2) * -1)[:, 0]
    world_point[:, 0] = (np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2) * -1)[:, 1]
    world_point = world_point * grid_length
    world_point = world_point + start_place

    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 查找角点
    ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), None)
    if not ok: # 没有检测到角点，则返回False
        return False
        # 获取更精确的角点位置
    exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 获取外参
    _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)
    print('--------------------平移矩阵T-----------------')
    print(tvec) # 平移矩阵

    print('--------------------旋转矩阵R-----------------')
    real_r = cv2.Rodrigues(rvec) # 旋转矩阵
    rvec = real_r[0]
    print(rvec)

    # 将相机的平移矩阵和旋转矩阵保存成.npz文件
    save_path = 'camera_parameters'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.savez(os.path.join(save_path, 'extrinsic_parameters'), tvec=tvec, rvec=rvec)
    return True

def get_calibration_pic():
    cap = cv2.VideoCapture(0)
    i = 100
    while True:
        ret, frame = cap.read()
        if not ret: # 获取相机图片失败的话就重试
            cap = cv2.VideoCapture(0)
            i -= 1
            if i == 0: # 重试了i次还不行就返回None
                break
            continue
        # print(i)
        return frame
    return ''

if __name__ == '__main__':
    # 棋盘按期望好的位置放上去，打开摄像头，得到标定图片
    calibration_pic = get_calibration_pic()
    if isinstance(calibration_pic, str): # 没有连接上相机或者相机连接除了问题
        print('获取相机的视频流失败，请确保相机已正确连接上！')
    else:
        # 进行图片标定
        flag = calibration_photo(calibration_pic)
        if not flag:
            print('标定失败，请调整好相机，保证整个棋盘都在相机视野内，且不会有太大偏斜角度！')
        else:
            print('标定成功！')