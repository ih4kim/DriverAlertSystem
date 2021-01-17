import cv2
import sys
import pdb
import time
import EyeIsolation
import ClosedEyeDetection
#import SpraySystem
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont

cap = cv2.VideoCapture(0)

#create a thread separate from the main loop
#this thread is responsible for retreiving frames from the webcam and 
#turning it into a format for PyQt and also calling the eyeIsolation and
#closedEyeDetection functions
#emits a signal indicating whether eyes are open/closed
class Thread(QThread):
    #define the signals to emit
    changePixmapMainCamera = pyqtSignal(QImage)
    changePixmapEye1 = pyqtSignal(QImage)
    changePixmapEye2 = pyqtSignal(QImage)
    changeTextEyeStatus = pyqtSignal(str)

    def convertToQT(self, image):
        height, width = image.shape
        #bytesPerLine = width
        ConvertToQTImage = QImage(image.data.tobytes(), width, height, width,QImage.Format_Grayscale8)
        scaled = ConvertToQTImage.scaled(width, height, Qt.KeepAspectRatio)
        return scaled
    
    def convertToNOTGRAYQT(self, image):
       # height, width = image.shape
        height = image.shape[1]
        width = image.shape[0]
        #bytesPerLine = width
        # ConvertToQTImage = QImage(image.data.tobytes(), width, height, width,QImage.Format_BGR888)
        # scaled = ConvertToQTImage.scaled(width, height, Qt.KeepAspectRatio)
        someosme = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # flippedImage = cv2.flip(image, 1)
        flippedImage = cv2.flip(someosme, 1)

        convertToQtFormat = QImage(flippedImage.data.tobytes(), flippedImage.shape[1], flippedImage.shape[0], QImage.Format_BGR888)
        scaled = convertToQtFormat.scaled(width, height, Qt.KeepAspectRatio)
        return scaled
        
    #called when start is called on QThread
    def run(self):
        startTimer = False
        eyesClosed = True
        model = ClosedEyeDetection.create_model()
        while True:
            ret, frame = cap.read()
            if ret:
                notGrayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eyeImages =  EyeIsolation.isolateEye(grayImage)
                eyesQT = []
                for eye in eyeImages:
                    eyesQT.append(self.convertToQT(eye))
                mainCamera = self.convertToNOTGRAYQT(notGrayImage)
                self.changePixmapMainCamera.emit(mainCamera)
                if (len(eyesQT)>0):
                    self.changePixmapEye1.emit(eyesQT[0])
                if (len(eyesQT)>1):
                    self.changePixmapEye2.emit(eyesQT[1])
                    tooFar = False
                    for i, eyeImage in enumerate(eyeImages):
                        if(eyeImage.shape <= (70,70)):
                            print("Too far !!!!")
                            tooFar = True
                        else: 
                            eyeImages[i] = ClosedEyeDetection.crop_center(eyeImages[i], 70, 70)
                            eyeImages[i] = eyeImages[i] / 255.0
                    #emit appropriate signals
                    #when a signal is emitted (changeTextEyeStatus is a signal), then whichever 
                    #function is connected to the signal in the main loop (self.setEyeStatusLabel)
                    #gets called with the signal value('Eyes Closed') as a function param
                    if(not tooFar):
                        eyesClosed = ClosedEyeDetection.eyeClosed(model, eyeImages)
                        if (ClosedEyeDetection.eyeClosed(model, eyeImages) == True and startTimer == False):
                            self.changeTextEyeStatus.emit('Eyes Closed')
                            initTime = time.perf_counter()
                            startTimer = True
                        elif (ClosedEyeDetection.eyeClosed(model, eyeImages) == True and startTimer == True):
                            timePassed = time.perf_counter() - initTime
                            self.changeTextEyeStatus.emit('Eyes Closed')
                            if (timePassed > 5):
                                SpraySystem.spray()
                                startTimer = False
                        elif (ClosedEyeDetection.eyeClosed(model, eyeImages) == False):
                            self.changeTextEyeStatus.emit('Eyes Opened')
                            startTimer = False

#this is the main window which inherits from QWidget
#responsible for setting up the main window and receiving signals emitted from the other thread
#the received signals are the frames (webcam images) and eye status; then this thread displays these
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Driver Alert System'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    #slots for the signals emitted in the QThread thread
    @pyqtSlot(QImage)
    def setMainCameraImage(self, image):
        self.MainCamera.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot(QImage)
    def setEye1Image(self, image):
        self.Eye1.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setEye2Image(self, image):
        self.Eye2.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str)
    def setEyeStatusLabel(self, status):
        if(status=='Eyes Closed'):
            self.EyeStatus.setText('STATUS: ASLEEP! WAKE UP!')
            self.EyeStatus.setStyleSheet("color: red;")
        else:            
            self.EyeStatus.setText('STATUS: GOOD (BOTH EYES OPEN)')
            self.EyeStatus.setStyleSheet("color: black;")

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1370, 1070)
        self.setStyleSheet("background-color: #F8F4E4;") 
        
        #creating labels (widgets)
        self.headerImage = QLabel(self)
        self.headerImage.setPixmap(QPixmap("noDrowsyDrivingHeader.png"))
        self.headerImage.move(100, 100)

        self.footerImage = QLabel(self)
        self.footerImage.setPixmap(QPixmap("noDrowsyDrivingFooter.png"))
        self.footerImage.move(100, 750)

        self.EyeStatus = QLabel(self)
        self.EyeStatus.move(400, 700)

        self.MainCamera = QLabel(self)
        self.MainCamera.move(450, 300)

            # self.Eye1 = QLabel(self)
            # self.Eye1.move(1000, 120)
            # self.Eye2 = QLabel(self)
            # self.Eye2.move(1000, 220)

        self.MainCamera.resize(640, 275)
        self.EyeStatus.resize(700, 100)
            # self.Eye1.resize(100, 100)
            # self.Eye2.resize(100, 100)

        self.EyeStatus.setFont(QFont('Arial', 15)) 
        self.EyeStatus.setStyleSheet("color: white;")

        th = Thread(self)
        #connect the signals to the slots
        th.changePixmapMainCamera.connect(self.setMainCameraImage)
        th.changeTextEyeStatus.connect(self.setEyeStatusLabel)
            # th.changePixmapEye1.connect(self.setEye1Image)
            # th.changePixmapEye2.connect(self.setEye2Image)
        th.start()
        self.show()

#initialize main window
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
    cap.release()
    cv2.destroyAllWindows()