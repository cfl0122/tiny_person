#coding:utf-8
import numpy as np
import os,math

# 内参矩阵
M1 = np.array([[509.0867, 0, 311.2543],
               [0, 511.3800, 228.1676],
               [0, 0, 1]])

# 外参矩阵
# M2 = np.array([[-0.01859716, -0.99940274, -0.02912563, 4.97160084],
#                [-0.61004235, 0.03442207, -0.79162078, 156.03610821],
#                [0.79215054, 0.00304597, -0.61031815, 118.4500752]])
path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'camera_parameters')
fpath = os.path.join(path , 'extrinsic_parameters.npz')
print(fpath)
assert os.path.exists(fpath), '请先进行外参标定'

# 读取相机外参

with np.load(fpath) as X:
    rvec, tvec = [X[i] for i in ('rvec', 'tvec')]
M2 = np.hstack((rvec, tvec)) # tvec在rvec的基础上扩充列

# 内参矩阵与外参矩阵的乘积
M = M1.dot(M2)

def get_X_Y(u, v, arr=M):
    a = arr[0][0]
    b = arr[0][1]
    c = arr[0][2]
    d = arr[0][3]
    e = arr[1][0]
    f = arr[1][1]
    g = arr[1][2]
    h = arr[1][3]
    i = arr[2][0]
    j = arr[2][1]
    k = arr[2][2]
    l = arr[2][3]

    A = u * i - a
    B = u * j - b
    C = u * l - d
    D = v * i - e
    E = v * j - f
    F = v * l - h

    Y = (C * D - A * F) / (A * E - B * D)
    X = (-B * Y - C) / A
    return X, Y

def get_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5




def world_info(top, left, bottom, right):

    print('左下角坐标：(%d, %d)' % (left, bottom))
    world_point1 = get_X_Y(left, bottom)
    dis1 = get_distance(world_point1, (0, 0))
    theta1 = math.atan(world_point1[1] / world_point1[0]) * 180 / math.pi
    print('左下角距离原点的距离为：%.2fmm, 偏转角：%.2f度' % (dis1, theta1))

    print('右下角坐标：(%d, %d)' % (right, bottom))
    world_point2 = get_X_Y(right, bottom)
    dis2 = get_distance(world_point2, (0, 0))
    theta2 = math.atan(world_point2[1] / world_point2[0]) * 180 / math.pi
    print('右下角距离原点的距离为：%.2fmm, 偏转角：%.2f度' % (dis2, theta2))

    print('中点坐标：(%d, %d)' % ((left + right) // 2, bottom))
    world_point3 = get_X_Y((left + right) // 2, bottom)
    dis3 = get_distance(world_point3, (0, 0))
    theta3 = math.atan(world_point3[1] / world_point3[0]) * 180 / math.pi
    print('中点距离原点的距离为：%.2fmm, 偏转角：%.2f度' % (dis3, theta3))


    # 根据余弦定理c ** 2 = a ** 2 + b ** 2 - 2 * a * b * cosC
    print(abs(theta1 - theta2))
    length = math.sqrt(dis1 ** 2 + dis2 ** 2 - 2 * dis1 * dis2 * math.cos(abs(theta1 - theta2) * math.pi / 180))
    print('物体的大概长度为：%.2fmm' % length)
    
    
    return world_point1,world_point2,world_point3,dis3,length
