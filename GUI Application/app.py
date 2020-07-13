from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from utils import *
import cv2

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):            
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(447, 492)
        MainWindow.setMaximumSize(QtCore.QSize(447, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.xrayImage = QtWidgets.QLabel(self.centralwidget)
        self.xrayImage.setGeometry(QtCore.QRect(20, 30, 200, 251))
        self.xrayImage.setText("")
        self.xrayImage.setObjectName("xrayImage")
        
        self.predictedLabel = QtWidgets.QLabel(self.centralwidget)
        self.predictedLabel.setGeometry(QtCore.QRect(40, 280, 361, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.predictedLabel.setFont(font)
        self.predictedLabel.setObjectName("predictedLabel")

        self.browseImageBtn = QtWidgets.QPushButton(self.centralwidget)
        self.browseImageBtn.setGeometry(QtCore.QRect(250, 20, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.browseImageBtn.setFont(font)
        self.browseImageBtn.setObjectName("browseImageBtn")
        self.browseImageBtn.clicked.connect(self.browseImage)

        self.predictBtn = QtWidgets.QPushButton(self.centralwidget)
        self.predictBtn.setGeometry(QtCore.QRect(250, 80, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.predictBtn.setFont(font)
        self.predictBtn.setObjectName("predictBtn")
        self.predictBtn.clicked.connect(self.prediction)

        
        self.probLabel = QtWidgets.QLabel(self.centralwidget)
        self.probLabel.setGeometry(QtCore.QRect(40, 320, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.probLabel.setFont(font)
        self.probLabel.setObjectName("probLabel")

        self.covidPositive_Prob = QtWidgets.QLabel(self.centralwidget)
        self.covidPositive_Prob.setGeometry(QtCore.QRect(40, 360, 361, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.covidPositive_Prob.setFont(font)
        self.covidPositive_Prob.setObjectName("covidPositive_Prob")
        self.covidPositive_Prob.setWordWrap(True)


        self.covidNegative_Prob = QtWidgets.QLabel(self.centralwidget)
        self.covidNegative_Prob.setGeometry(QtCore.QRect(40, 400, 361, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.covidNegative_Prob.setFont(font)
        self.covidNegative_Prob.setObjectName("covidNegative_Prob")
        self.covidNegative_Prob.setWordWrap(True)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "COVID-19 Detector"))
        self.predictedLabel.setText(_translate("MainWindow", "PREDICTION:"))
        self.browseImageBtn.setText(_translate("MainWindow", "Browse Image"))
        self.predictBtn.setText(_translate("MainWindow", "Analyse"))
        self.probLabel.setText(_translate("MainWindow", "Probabilities"))
        self.covidPositive_Prob.setText(_translate("MainWindow", "COVID-19 +ve:"))
        self.covidNegative_Prob.setText(_translate("MainWindow", "COVID-19 -ve:"))

    def browseImage(self):
        fm = QtWidgets.QFileDialog.getOpenFileName(None,"OpenFile")
        filename = fm[0]
        self.image = cv2.imread(filename)        
        self.xrayImage.setPixmap(QtGui.QPixmap(filename))
        self.xrayImage.setScaledContents(True)
    
    def prediction(self):
        self.image = cv2.resize(self.image, (image_size,image_size))
        print("Analysis....")        
        try:
            label, probabilities = predict(self.image)   #("COVID",[0.98,0.02])     
            self.predictedLabel.setText("PREDICTION: "+label)
            #print(probabilities)
            self.covidPositive_Prob.setText("COVID-19 +ve: " + str(probabilities[0]))
            self.covidNegative_Prob.setText("COVID-19 -ve: " + str(probabilities[1]))
            print("Analysis done")
        except:
            msgError = QtWidgets.QMessageBox()
            msgError.setIcon(QtWidgets.QMessageBox.Critical)
            msgError.setWindowTitle("Error")
            msgError.setText("Oops!! Error")
            msgError.exec_()
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())