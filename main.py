import pkg.processing as pr
import cv2
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtCore

form_class = uic.loadUiType("UI.ui")[0]

def main():
    img1 = cv2.imread('lena.jpg')
    img2 = cv2.imread('abcdef.bmp')
    output_img = pr.laplacianOfGaussianEdge(img1)

    #output_img_resize = cv2.resize(output_img, (0, 0), fx=2, fy=2)

    cv2.imshow('image', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Photoshop(QMainWindow, form_class):
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        self.actionOpen.triggered.connect(self.fileOpen)
        self.actionUndo.triggered.connect(self.undo)

        #Rotate
        self.actionClockwise_90.triggered.connect(self.rotateClockwise)
        self.actionCounter_Clockwise_90.triggered.connect(self.rotateCounterClockwise)
        self.action180_deg.triggered.connect(self.rotate180)
        self.actionHorizontal_Flip.triggered.connect(self.horizontalFlip)
        self.actionVertical_Flip.triggered.connect(self.verticalFlip)

        #Pixel Processing
        self.actionAdd.triggered.connect(self.add)
        self.actionSubtract.triggered.connect(self.subtract)
        self.actionMultiplication.triggered.connect(self.multiplication)

        #Bit Processing
        self.actionAND.triggered.connect(self.AND)
        self.actionOR.triggered.connect(self.OR)
        self.actionNOT.triggered.connect(self.NOT)

        #Filtering
        self.actionMean_Filtering.triggered.connect(self.meanFiltering)
        self.actionMedian_Filtering.triggered.connect(self.medianFiltering)
        self.actionGaussian_Filtering.triggered.connect(self.gaussianFiltering)
        self.actionConservative_Smoothing.triggered.connect(self.conservativeSmoothing)
        self.actionUnsharp_Filtering.triggered.connect(self.unsharpFiltering)

        #Edge Detect
        self.actionRoberts_cross_edge.triggered.connect(self.robertsCrossEdge)
        self.actionSobel_edge.triggered.connect(self.sobelEdge)
        self.actionPrewitt_edge.triggered.connect(self.prewittEdge)
        self.actionCanny_edge.triggered.connect(self.cannyEdge)
        self.actionLaplacian_edge.triggered.connect(self.laplacianEdge)
        self.actionLaplacian_of_Gaussian_edge.triggered.connect(self.laplacianOfGaussianEdge)

        #Morphology
        self.actionDilation.triggered.connect(self.dilation)
        self.actionErosion.triggered.connect(self.erosion)
        self.actionOpening.triggered.connect(self.opening)
        self.actionClosing.triggered.connect(self.closing)

    #File
    def fileOpen(self):
        global image
        fname = QFileDialog.getOpenFileName(self, 'Open File', '', 'All File(*);; Image File(*.png *.jpg *.bmp')
        if fname[0]:
            pixmap = QtGui.QPixmap(fname[0])
            image = cv2.imread(fname[0])
            self.label.setPixmap(pixmap)

    def undo(self):
        global image
        temp = image
        image = cv2.imread('temp2.jpg')
        pixmap = QtGui.QPixmap('temp2.jpg')
        cv2.imwrite('temp2.jpg', temp)
        cv2.imwrite('temp.jpg', image)
        self.label.setPixmap(pixmap)

    #Rotate
    def rotateClockwise(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.rotateClockwise(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def rotateCounterClockwise(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.rotateCounterClockwise(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def rotate180(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.rotate180Deg(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def horizontalFlip(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.horizontalFlip(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def verticalFlip(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.verticalFlip(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    #Pixel Processing
    def add(self):
        global image
        fname = QFileDialog.getOpenFileName(self, 'Open File', '', 'All File(*);; Image File(*.png *.jpg *.bmp')
        if fname[0]:
            image2 = cv2.imread(fname[0])
            cv2.imwrite('temp2.jpg', image)
            image = pr.add(image, image2)
            cv2.imwrite('temp.jpg', image)
            pixmap = QtGui.QPixmap('temp.jpg')
            self.label.setPixmap(pixmap)

    def subtract(self):
        global image
        fname = QFileDialog.getOpenFileName(self, 'Open File', '', 'All File(*);; Image File(*.png *.jpg *.bmp')
        if fname[0]:
            image2 = cv2.imread(fname[0])
            cv2.imwrite('temp2.jpg', image)
            image = pr.subtract(image, image2)
            cv2.imwrite('temp.jpg', image)
            pixmap = QtGui.QPixmap('temp.jpg')
            self.label.setPixmap(pixmap)

    def multiplication(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.multiplication(image, 2.0)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    #Bit Processing
    def AND(self):
        global image
        fname = QFileDialog.getOpenFileName(self, 'Open File', '', 'All File(*);; Image File(*.png *.jpg *.bmp')
        if fname[0]:
            image2 = cv2.imread(fname[0])
            cv2.imwrite('temp2.jpg', image)
            image = pr.pixelAND(image, image2)
            cv2.imwrite('temp.jpg', image)
            pixmap = QtGui.QPixmap('temp.jpg')
            self.label.setPixmap(pixmap)

    def OR(self):
        global image
        fname = QFileDialog.getOpenFileName(self, 'Open File', '', 'All File(*);; Image File(*.png *.jpg *.bmp')
        if fname[0]:
            image2 = cv2.imread(fname[0])
            cv2.imwrite('temp2.jpg', image)
            image = pr.pixelOR(image, image2)
            cv2.imwrite('temp.jpg', image)
            pixmap = QtGui.QPixmap('temp.jpg')
            self.label.setPixmap(pixmap)

    def NOT(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.pixelComplement(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    #Filtering
    def meanFiltering(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.meanFiltering(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def medianFiltering(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.medianFiltering(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def gaussianFiltering(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.gaussianFiltering(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def conservativeSmoothing(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.conservativeSmoothing(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def unsharpFiltering(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.unsharpFiltering(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    #Edge Detect
    def robertsCrossEdge(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.robertsCrossEdge(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def sobelEdge(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.sobelEdge(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def prewittEdge(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.prewittEdge(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def cannyEdge(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.cannyEdge(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def laplacianEdge(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.laplacianEdge(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def laplacianOfGaussianEdge(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.laplacianOfGaussianEdge(image)
        cv2.imwrite('temp.jpg', image)
        image = cv2.imread('temp.jpg')
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    #Morphology
    def dilation(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.dilation(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def erosion(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.erosion(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def opening(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.opening(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

    def closing(self):
        global image
        cv2.imwrite('temp2.jpg', image)
        image = pr.closing(image)
        cv2.imwrite('temp.jpg', image)
        pixmap = QtGui.QPixmap('temp.jpg')
        self.label.setPixmap(pixmap)

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = Photoshop()
    myWindow.show()
    app.exec_()