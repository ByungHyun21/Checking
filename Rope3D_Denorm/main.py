import sys
import cv2
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.opengl import GLViewWidget, GLMeshItem

folder = "data1"

def load_calib():
    #read txt
    with open(f"{folder}/calib.txt") as f:
        lines = f.readlines()
    #parse
    calib = [line.split(" ") for line in lines]
    calib = calib[0][1:]
    calib = np.array(calib, dtype=np.float32).reshape(3, 4)
    return calib

def load_denorm():
    #read txt
    with open(f"{folder}/denorm.txt") as f:
        lines = f.readlines()
    #parse
    denorm = [line.split(" ") for line in lines]
    denorm = np.array(denorm[0], dtype=np.float32)
    return denorm

def load_label():
    #read txt
    with open(f"{folder}/label.txt") as f:
        lines = f.readlines()
    #parse
    labels = [line.split(" ") for line in lines]
    for i in range(len(labels)):
        labels[i] = np.array(labels[i][1:], dtype=np.float32).tolist()
    return labels


calib = load_calib()
denorm = load_denorm()
label = load_label()

rot_d = np.array([
    [1, 0, 0],
    [0, -denorm[1], +denorm[2]],
    [0, -denorm[2], -denorm[1]]
])

app = QtWidgets.QApplication(sys.argv)  # QApplication을 QtWidgets에서 가져옵니다.
view = GLViewWidget()
view.show()
view.setWindowTitle('3D Box Viewer')
view.setCameraPosition(distance=50)

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
    vertices = (rot_d @ r @ vertices).T + t
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

for i in range(len(label)):
    # parse KITTI label
    dimensions = label[i][7:10]
    location = label[i][10:13]
    rotation_y = label[i][13]
    
    r = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    
    t = np.array(location)
    s = np.array(dimensions)[[2, 0, 1]].reshape(3, 1)
    vertice, face = create_box_mesh(r, t, s)
    meshdata = pg.opengl.MeshData(vertexes=vertice, faces=face)
    item = GLMeshItem(meshdata=meshdata, color=color, edgeColor=(0, 1, 0, 1), drawEdges=True, smooth=False)
    view.addItem(item)

# let's check what is denorm doing
# maybe denorm is plain parameter (a, b, c, d): ax + by + cz + d = 0
a, b, c, d = denorm
md = 10
x = np.array([md, md, -md, -md])
y = np.array([md, -md, md, -md])
Z = (-a * x - b * y - d) / c

vertexes = np.array([
    [md, md, Z[0]],
    [md, -md, Z[1]],
    [-md, md, Z[2]],
    [-md, -md, Z[3]]
], dtype=np.float32)
face = np.array([
    [0, 1, 2],
    [1, 3, 2]
])

mesh = pg.opengl.MeshData(vertexes=vertexes, faces=face)
item = GLMeshItem(meshdata=mesh, color=(1, 1, 1, 0.3), edgeColor=(1, 1, 1, 0.3), drawEdges=True, smooth=False)
view.addItem(item)

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()