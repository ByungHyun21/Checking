import sys
import cv2
import numpy as np
import json

from scipy.spatial.transform import Rotation as R

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.opengl import GLViewWidget, GLMeshItem

# data3_ours : aihub
# data4_ours : aihub
# data5_ours : rope3d
# data6_ours : rope3d

# folder = "data4_ours"
# d = 9

folder = "data6_ours"
d = 8

def load_cam():
    path = f"{folder}/cam.jpg"
    image = cv2.imread(path)
    
    with open(f"{folder}/cam.json", "r") as f:
        label = json.load(f)
    
    fx = label['intrinsic']['fx']
    fy = label['intrinsic']['fy']
    cx = label['intrinsic']['cx']
    cy = label['intrinsic']['cy']
    label['intrinsic'] = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    ext_rot = np.array(label['extrinsic']['rotation'], dtype=np.float32).reshape(3, 3)
    tx = label['extrinsic']['translation']['x']
    ty = label['extrinsic']['translation']['y']
    tz = label['extrinsic']['translation']['z']
    ext_trans = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)
    label['extrinsic'] = np.concatenate([ext_rot, ext_trans], axis=1)
    
    return image, label

def load_lidar():
    path = f"{folder}/lidar.bin"
    lidar = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    
    with open(f"{folder}/lidar.json", "r") as f:
        label = json.load(f)
    
    ext_rot = np.array(label['extrinsic']['rotation'], dtype=np.float32).reshape(3, 3)
    tx = label['extrinsic']['translation']['x']
    ty = label['extrinsic']['translation']['y']
    tz = label['extrinsic']['translation']['z']
    ext_trans = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)
    label['extrinsic'] = np.concatenate([ext_rot, ext_trans], axis=1)
    
    return lidar, label

def rotationMatrixToEulerAngles(matrix: np.array, type: str):
    # matrix: 3x3 rotation matrix
    # type: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
    # return: euler angles
    r = R.from_matrix(matrix)
    return r.as_euler(type, degrees=False)

def eulerAnglesToRotationMatrix(rot: np.array, type: str):
    # rot: radian euler angles
    # type: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
    # return: 3x3 rotation matrix
    r = R.from_euler(type, rot, degrees=False)
    return r.as_matrix()
  
image, cam_label = load_cam()
# lidar, lidar_label = load_lidar()

norm = np.array([0.0, 0.0, 1.0], dtype=np.float32).reshape(3, 1)
norm = cam_label['extrinsic'][:3, :3] @ norm
# norm = -norm





app = QtWidgets.QApplication(sys.argv)  # QApplication을 QtWidgets에서 가져옵니다.
view = GLViewWidget()
view.show()
view.setWindowTitle('3D Box Viewer')
view.setCameraPosition(distance=50)

# let's check what is denorm doing
# maybe denorm is plain parameter (a, b, c, d): ax + by + cz + d = 0
a, b, c, d = norm[0][0], norm[1][0], norm[2][0], d
md = 30


x = np.array([md, md, -md, -md])
z = np.array([md, -md, md, -md])
Y = (-a * x - c * z - d) / (b + 1e-6)
vertexes = np.array([
    [md, Y[0], md],
    [md, Y[1], -md],
    [-md, Y[2], md],
    [-md, Y[3], -md]
], dtype=np.float32)



face = np.array([
    [0, 1, 2],
    [1, 3, 2]
])

mesh = pg.opengl.MeshData(vertexes=vertexes, faces=face)
item = GLMeshItem(meshdata=mesh, color=(1, 1, 1, 0.3), edgeColor=(1, 1, 1, 0.3), drawEdges=True, smooth=False)
view.addItem(item)

xaxis = np.concatenate([[np.zeros(3)], [np.array([10, 0, 0])]])
yaxis = np.concatenate([[np.zeros(3)], [np.array([0, 10, 0])]])
zaxis = np.concatenate([[np.zeros(3)], [np.array([0, 0, 10])]])

view.addItem(pg.opengl.GLLinePlotItem(pos=xaxis, color=(1, 0, 0, 1), width=3))
view.addItem(pg.opengl.GLLinePlotItem(pos=yaxis, color=(0, 1, 0, 1), width=3))
view.addItem(pg.opengl.GLLinePlotItem(pos=zaxis, color=(0, 0, 1, 1), width=3))

# 색상 설정
color = (0, 0, 0, 0)  # 흰색, 투명도 설정

def create_box_mesh(r, t, s):
    # 8개의 정점 정의
    vertices = np.array([
        [1, 2, 1],
        [1, 2, -1],
        [1, 0, 1],
        [1, 0, -1],
        [-1, 2, 1],
        [-1, 2, -1],
        [-1, 0, 1],
        [-1, 0, -1]
    ], dtype=np.float32) * 0.5
    
    # 스케일, 회전, 평행 이동 적용
    vertices = s * vertices.T
    vertices = r @ vertices + t
    
    # 6개의 면을 정의하는 삼각형들
    faces = np.array([
        [0, 1, 2], [1, 3, 2],  # 앞
        [4, 5, 6], [5, 7, 6],  # 뒤
        [0, 1, 4], [1, 5, 4],  # 위
        [2, 3, 6], [3, 7, 6],  # 아래
        [0, 2, 4], [2, 6, 4],  # 왼쪽
        [1, 3, 5], [3, 7, 5]   # 오른쪽
    ])
    
    return vertices, faces

for obj in cam_label['objects']:
    box3d = obj['box3d']
    dimensions = np.array([box3d['size']['width'], box3d['size']['height'], box3d['size']['length']], dtype=np.float32)
    t = np.array([box3d['translation']['x'], box3d['translation']['y'], box3d['translation']['z']], dtype=np.float32).reshape(3, 1)
    r = np.array(box3d['rotation'], dtype=np.float32).reshape(3, 3)
    
    s = np.array(dimensions)[[2, 0, 1]].reshape(3, 1)
    vertice, face = create_box_mesh(r, t, s)
    meshdata = pg.opengl.MeshData(vertexes=vertice.T, faces=face)
    item = GLMeshItem(meshdata=meshdata, color=color, edgeColor=(0, 1, 0, 1), drawEdges=True, smooth=False)
    view.addItem(item)



if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()