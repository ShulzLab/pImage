# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 02:12:06 2020

@author: Timothe
"""

try :
    from skimage.draw import line_aa
    from skimage import measure,filters
    from scipy import signal
    from PyQt5.QtWidgets import QDialog, QDialogButtonBox
except ImportError as e:
    QDialog = e
import numpy as np
import cv2
import os,sys
from cv2 import VideoWriter, VideoWriter_fourcc
#
import pyprind

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])

sys.path.append(uppath(__file__, 2))




# def ImageCorrelation(Image1,Image2):
#     """
#     Calulate the 2D correlation of two images together.

#     Parameters
#     ----------
#     Image1 : np.ndarray (2D)
#         First image.
#     Image2 : np.ndarray (2D)
#         Second image.

#     Returns
#     -------
#     cor : TYPE
#         DESCRIPTION.

#     """
#     cor = signal.correlate2d (Image1, Image2)
#     return cor




# def RandImage(X,Y,bindepth = 8):
#     """
#     Generate a 2D numpy ndarray of dimension 1 and 2 corresponding to X and Y size of the image.
#     The values of the pixels in that random array ranges from 0 to the max value of the desired bit depth. (eg. bindepth = 8 : maxvalue = 256 etc.)

#     Parameters
#     ----------
#     X : int
#         1st dimension size.
#     Y : int
#         2nd dimension size.
#     bindepth : int, optional
#         Bin depth constraining the pixels max value (eg : 8bit = maxvalue 256). The default is 8.
#         Please note thtat the minimum value is 0 whatever the bindepth value is.

#     Returns
#     -------
#     numpy.ndarray (2D).

#     """
#     maxval = max_value_bits(bindepth)
#     return np.random.rand(X,Y)*maxval

# def RandVideo(X,Y,Time,bindepth = 8):
#     """

#     Generate a 3D numpy ndarray of dimension 1 and 2 corresponding to X and Y size of the image, and 3rd dimension as time (number of frames).
#     The values of the pixels in that random array ranges from 0 to the max value of the desired bit depth. (eg. bindepth = 8 : maxvalue = 256 etc.)

#     Parameters
#     ----------
#     X : int
#         1st dimension size
#     Y : int
#         2nd dimension size.
#     Time : 3nd dimension size.
#         DESCRIPTION.
#     bindepth : int, optional
#         Bin depth constraining the pixels max value (eg : 8bit = maxvalue 256). The default is 8.
#         Please note thtat the minimum value is 0 whatever the bindepth value is.

#     Returns
#     -------
#     numpy.ndarray (3D) with time as 3rd dimension

#     """
#     maxval = max_value_bits(bindepth)
#     return np.random.rand(X,Y,Time)*maxval







def max_value_bits(b):
    """
    Get maximum (unsigned) value of a given integer bit size variable.

    Parameters
    ----------
    b : int
        Number of bits (binary values) that are used to describe a putative variable.

    Returns
    -------
    max_value : int
        Maximum value that putative variable can hold (integer unsigned).

    """
    return (2 ** b) - 1

# def QuickHist(image, **kwargs):
#     """
#     Generate histograms of pixel values of an image (color or gray).

#     Parameters
#     ----------
#     image (numpy.ndarray):
#         2D or 3D np array with x,y as first two dimensions, and third dimension as RGB channels if color image.

#     **color (bool): default = False
#         Informs if the array is colored RBG layered data that needs to be averaged as gray to generate histogram
#         True : image should be a 3D numpy ndarray (eg.color image)
#         False : image should be a 2D numpy ndarray (eg.gray image)

#     **bindepth (int): default = 8
#         Maximum value of a pixel (8,12,16 bit depths for example)

#     **display (bool): default = True
#         True to have the function generate a plot.
#         False to have the function return a numpy 1D array containing the histogram values.

#     Returns
#     -------
#     None, or numpy.ndarray
#         If display is set to False, the function returns a numpy 1D array containing the histogram values..

#     """
#     import matplotlib.pyplot as plt
#     import cv2

#     # calculate mean value from RGB channels if presents and flatten to 1D array
#     color = kwargs.get("color", False)
#     if color :
#         vals = image.mean(axis=2).flatten()
#     else :
#         vals = image.flatten()

#     bindepth = kwargs.get("bindepth", 8)
#     maxval = max_value_bits(bindepth)

#     display = kwargs.get("display", True)
#     if display :
#         # plot histogram with 255 bins
#         b, bins, patches = plt.hist(vals, maxval)
#         plt.xlim([0,maxval])
#         plt.show()
#     else :
#         return np.histogram(vals,range = (0,maxval))

class _VideoVisualizeDialog(QDialog):
    import PyQt5.QtCore

    def __init__(self, VideoArray, parent=None, **kwargs):
        """
        Call this to open a Qt Gui to visualize images or videos with variable time/data context change.

        Parameters.

        ----------
        VideoArray : Video as a 3D array with x,y as fist dimensions and time as third dimension.
            DESCRIPTION.
        parent : Object type to bound this GUI to, optional. Do not use if you don't know the code structure (just ommit it)
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        A dictionnary depending of the mode and the user actions.

        """
        import LibrairieQtDataDive.widgets as visu

        from PyQt5.QtWidgets import QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QSlider
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap, QImage, qRgb

        super(_VideoVisualizeDialog, self).__init__(parent)

        title = kwargs.get("title" , "Is video valid ?")

        self.setWindowTitle(title)

        mode = kwargs.get("mode", "view")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        buttonBox = QDialogButtonBox(QBtn)
        buttonBox.accepted.connect(self.acceptv)
        buttonBox.rejected.connect(self.rejectv)

        self.VIDwidget = visu.Plot3DWidget(parent = self)
        self.VIDwidget.IAuto.setChecked(True)

        NoTrackerButton = QPushButton("No tracker")
        NoTrackerButton.pressed.connect(self.notrackerv)

        buttons = QGroupBox()

        self.layout = QHBoxLayout()

        if mode == "coords":

            self.VIDwidget.setlowCM(True)
            self.VIDwidget.SetData(VideoArray,**kwargs)

            SkipButton = QPushButton("Skip Session")
            SkipButton.pressed.connect(self.Skip)

            self.layout.addWidget(buttonBox)
            self.layout.addWidget(NoTrackerButton)
            self.layout.addWidget( SkipButton )
            self.layout.addWidget( QLabel(title) )

            trackerimg = kwargs.get("tracker_img",None)

            if trackerimg is not None :
                qimage = QImage(trackerimg.data, trackerimg.shape[1], trackerimg.shape[0], trackerimg.strides[0], QImage.Format_Indexed8)
                qimage.setColorTable([qRgb(i, i, i) for i in range(256)])
                pix = QPixmap(qimage)
                trackerimg_label = QLabel()
                trackerimg_label.setPixmap(pix)
                trackerimg_label.setMinimumSize(100, 100)
                self.layout.addWidget( trackerimg_label )

            self.cid2 = self.VIDwidget.canvas.fig.canvas.mpl_connect('button_press_event', self.ClickCoords)

        if mode == "view":

            self.VIDwidget.setlowCM(True)
            self.VIDwidget.SetData(VideoArray,**kwargs)
            self.layout.addWidget(buttonBox)
            self.layout.addWidget( QLabel(title) )

            self.cid2 = self.VIDwidget.canvas.fig.canvas.mpl_connect('button_press_event', self.ClickCoords)

        if mode == 'binarise':

            self.VIDwidget.setlowCM(True)
            self.VIDwidget.SetData(VideoArray,**kwargs)
            buttonBox.accepted.disconnect()
            buttonBox.accepted.connect(self.binarresultv)

            #binbutton = QPushButton("Update Bin.")
            #binbutton.pressed.connect(self.BinarizeChange)

            self.BinarizeSlider = QSlider(Qt.Horizontal, self)
            self.BinarizeSlider.setMaximum(254)
            self.BinarizeSlider.setMinimum(0)
            self.BinarizeSlider.valueChanged.connect(self.BinarizeChange)

            self.BinReadout = QLabel("-")

            self.VIDwidget.SupSlider.Slider.sliderReleased.connect(self.BinarizeChange)

            self.layout.addWidget(self.BinarizeSlider)
            self.layout.addWidget(QLabel("Binar. Threshold"))
            self.layout.addWidget(self.BinReadout)
            #self.layout.addWidget(binbutton)
            self.layout.addWidget(buttonBox)
            self.layout.addWidget( QLabel(title) )

            self.cid2 = self.VIDwidget.canvas.fig.canvas.mpl_connect('button_press_event', self.ClickCoords)


        if mode == 'full':


            self.VIDwidget.setlowCM(False)
            self.VIDwidget.SetData(VideoArray,**kwargs)
            self.layout.addWidget(buttonBox)
            self.layout.addWidget( QLabel(title) )


        buttons.setLayout(self.layout)

        self.returnDictionnary = { "retbool" : None }
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.VIDwidget)
        self.layout.addWidget(buttons)
        self.setLayout(self.layout)

        #self.cid2 = self.VIDwidget.canvas.fig.canvas.mpl_connect('button_press_event', self.ClickCoords)

        self.setWindowFlags( Qt.WindowStaysOnTopHint )
        #♣self.setAttribute(Qt.WA_TranslucentBackground)

        self.raise_()
        self.activateWindow()

        self.VIDwidget.UpdateFrame()

    def Skip(self):

        self.returnDictionnary.update({"skip" : True})
        self.returnDictionnary.update({"retbool" : 1 })
        self.close()

    def BinarizeChange(self):

        kwargs = self.VIDwidget.MakeKWARGS(False)
        self.VIDwidget.canvas.update_figure( binarize ( self.VIDwidget.RawDATA[:,:,self.VIDwidget.frame  ] , self.BinarizeSlider.value()  ) , **kwargs)
        self.BinReadout.setText(str(self.BinarizeSlider.value()))

    def notrackerv(self):
        self.returnDictionnary.update({"trackerfound" : False, "retbool" : 1})
        self.close()

    def binarresultv(self):
        self.returnDictionnary.update({"theshold" : self.BinarizeSlider.value() })
        self.returnDictionnary.update({"retbool" : 1 })
        self.close()

    def acceptv(self):
        self.returnDictionnary.update({"retbool" : 1 })
        self.close()

    def rejectv(self):
        self.returnDictionnary.update({"retbool" : 0 })
        self.close()
        #QtCore.QCoreApplication.instance().quit()

    def Popup(self):
        from PyQt5 import QtWidgets
        self.eta1, ok = QtWidgets.QInputDialog.getDouble(self,
                "Change of variable", "Rate (type 1):", 0.1, 0, 1e8, 3)

    def Save(self):

        try :
            self.returnDictionnary.update({'frame' : self.VIDwidget.SupSlider.Slider.value()})
            return self.returnDictionnary
        #{'x' : self.ix, 'y' : self.iy, 'frame' : self.VIDwidget.SupSlider.Slider.value(), "retbool" : self.returnBool, "trackerfound" : self.trackerfound }
        #(self.ix, self.iy, self.VIDwidget.SupSlider.Slider.value())
        except :
            return None

    def ClickCoordinates(self,Bool=True):

        if Bool:
            #print("setting up")
            self.cid2 = self.VIDwidget.canvas.fig.canvas.mpl_connect('button_press_event', self.ClickCoords)
        else :
            #print("disable")
            self.VIDwidget.canvas.fig.canvas.mpl_disconnect(self.cid2)

    def ClickCoords(self,event):
        ix,iy = event.xdata, event.ydata
        self.returnDictionnary.update({"x" : ix, "y" : iy , "trackerfound" : True, "retbool" : 1})
        #self.ClickCoordinates(False)
        self.VIDwidget.canvas.fig.canvas.mpl_disconnect(self.cid2)
        #print(self.ix, self.iy)
        self.close()


def VideoDialog(VideoArray,**kwargs):
    """
    test
    """
    from PyQt5 import QtWidgets
    qApp = QtWidgets.QApplication(sys.argv)
    dlg = _VideoVisualizeDialog(VideoArray,**kwargs)
    Value = dlg.exec_()
    coords = dlg.Save()
    del qApp, dlg
    if coords is not None :
        return coords
    else :
        return Value


if __name__ == "__main__":

    img = (np.random.rand(20,20,20)*254).astype(np.uint8)
    returnval = VideoDialog(img,mode = "coords")
    print(returnval)