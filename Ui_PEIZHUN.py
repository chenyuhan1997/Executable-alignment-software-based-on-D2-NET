# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Administrator\Desktop\装甲目标识别项目\配准\Project\PEIZHUN.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(775, 710)
        MainWindow.setMinimumSize(QtCore.QSize(775, 710))
        MainWindow.setMaximumSize(QtCore.QSize(775, 710))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 10, 754, 648))
        self.widget.setObjectName("widget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label_one = QtWidgets.QLabel(self.widget)
        self.label_one.setMinimumSize(QtCore.QSize(360, 270))
        self.label_one.setMaximumSize(QtCore.QSize(360, 270))
        self.label_one.setStyleSheet("border-width: 1px;border-style: solid;border-color: rgb(0,0,0);background-color: rgb(255,245,238);")
        self.label_one.setObjectName("label_one")
        self.verticalLayout.addWidget(self.label_one)
        self.pushButton_one = QtWidgets.QPushButton(self.widget)
        self.pushButton_one.setObjectName("pushButton_one")
        self.verticalLayout.addWidget(self.pushButton_one)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.label_two = QtWidgets.QLabel(self.widget)
        self.label_two.setMinimumSize(QtCore.QSize(360, 270))
        self.label_two.setMaximumSize(QtCore.QSize(360, 270))
        self.label_two.setStyleSheet("border-width: 1px;border-style: solid;border-color: rgb(0,0,0);background-color: rgb(255,245,238);")
        self.label_two.setObjectName("label_two")
        self.verticalLayout_2.addWidget(self.label_two)
        self.pushButton_two = QtWidgets.QPushButton(self.widget)
        self.pushButton_two.setObjectName("pushButton_two")
        self.verticalLayout_2.addWidget(self.pushButton_two)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.addWidget(self.label_6)
        self.out_label = QtWidgets.QLabel(self.widget)
        self.out_label.setMinimumSize(QtCore.QSize(750, 270))
        self.out_label.setMaximumSize(QtCore.QSize(750, 270))
        self.out_label.setStyleSheet("border-width: 1px;border-style: solid;border-color: rgb(0,0,0);background-color: rgb(255,245,238);")
        self.out_label.setObjectName("out_label")
        self.verticalLayout_3.addWidget(self.out_label)
        self.get_start = QtWidgets.QPushButton(self.widget)
        self.get_start.setObjectName("get_start")
        self.verticalLayout_3.addWidget(self.get_start)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 775, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "配准展示"))
        self.label_3.setText(_translate("MainWindow", "预配准图1"))
        self.label_one.setText(_translate("MainWindow", "图1"))
        self.pushButton_one.setText(_translate("MainWindow", "选择"))
        self.label_4.setText(_translate("MainWindow", "预配准图2"))
        self.label_two.setText(_translate("MainWindow", "图1"))
        self.pushButton_two.setText(_translate("MainWindow", "选择文件"))
        self.label_6.setText(_translate("MainWindow", "配准结果"))
        self.out_label.setText(_translate("MainWindow", "图1"))
        self.get_start.setText(_translate("MainWindow", "配准"))
