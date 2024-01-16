
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import *
import cv2
import numpy as np
from general.resize_keep import resize_keep_aspectratio
import filetype

def open_file(parent,outLabel,outSource):#打开文件
        
        image='0'
        source = QFileDialog.getOpenFileName(parent, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                           "*.jpg *.png)")
        
        
        outSource[0] = source[0]#写入文件路径
        print('outSource',outSource)
        kind =filetype.guess(source[0])
        if kind==None:
            return
        
        if kind.mime[0:5]=='video':#如果是视频
            cap = cv2.VideoCapture(source[0])
            cap.set(1, 1)  # 取它的第一帧
            rval, frame = cap.read()  # 如果rval为False表示这个视频有问题，为True则正常
            image=frame
            b,g,r = cv2.split(image)			#分别提取B、G、R通道
            image = cv2.merge([r,g,b])	#重新组合为R、G、B
            cap.release()
        if kind.mime[0:5]=='image':#如果是图片
            image=cv2.imdecode(np.fromfile(source[0], dtype=np.uint8), flags=cv2.IMREAD_COLOR)
            b,g,r = cv2.split(image)			#分别提取B、G、R通道
            image = cv2.merge([r,g,b])	#重新组合为R、G、B
      
        file_set(image,outLabel)

def file_set(image,outLabel):
        dst_size=(outLabel.height(),outLabel.width())
        image = resize_keep_aspectratio(image,dst_size)#输出的是全0矩阵
        xs=image.shape[1]*3#防止倾斜变形
        img = QImage(image.data, image.shape[1], image.shape[0],xs, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        outLabel.setPixmap(pixmap)