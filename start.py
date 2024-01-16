from Ui_PEIZHUN import Ui_MainWindow
import cv2, sys, time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap,QPainter,QImageReader
from PyQt5.QtCore import *
from general.file_open import open_file,file_set
from cnnmatching import match

class Demo(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):

        super(Demo, self).__init__(parent)
        self.setupUi(self)
        self.source1=['']
        self.source2=['']
       
        self.pushButton_one.clicked.connect(lambda:self.S_open_file(self.label_one,self.source1))#给可见光绑定本地资源 
        self.pushButton_two.clicked.connect(lambda:self.S_open_file(self.label_two,self.source2))#给可见光绑定本地资源 
        self.get_start.clicked.connect(lambda:self.start_match(self.source1[0],self.source2[0]))#绑定配准函数
   
   
    def S_open_file(self,outLabel,source):#打开文件

            open_file(self, outLabel, source)
    
    def start_match(self,source1,source2):
         if source1 !='' and source2!='':
            aaa = match(source1,source2)
            if aaa=='good':
                print('ggg')
            # time.sleep(0.5)
                image = cv2.imread('matching_result.png')
                b,g,r = cv2.split(image)			#分别提取B、G、R通道
                image = cv2.merge([r,g,b])	#重新组合为R、G、B
                file_set(image,self.out_label) 
            



if __name__ == '__main__':


    app = QApplication(sys.argv)
    ui=Demo()
    ui.show()
    sys.exit(app.exec_())