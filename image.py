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


def Empty_img(X,Y):
    """
    Generate a black image of a given dimension with a white cross on the middle, to use as "no loaded image" user readout.

    Parameters
    ----------
    X : int
        Shape for first dimension on the generated array (X)
    Y : int
        Shape for second dimension on the generated array (Y).

    Returns
    -------
    img : numpy.ndarray
        Blank image with white cross, of [X,Y] shape.

    """
    img = np.zeros((X, Y), dtype=np.uint8)
    rr, cc, val = line_aa(0, 0, X-1, Y-1)
    img[rr, cc] = val * 255
    rr, cc, val = line_aa(0, Y-1, X-1, 0)
    img[rr, cc] = val * 255

    return img

def ImageCorrelation(Image1,Image2):
    """
    Calulate the 2D correlation of two images together.

    Parameters
    ----------
    Image1 : np.ndarray (2D)
        First image.
    Image2 : np.ndarray (2D)
        Second image.

    Returns
    -------
    cor : TYPE
        DESCRIPTION.

    """
    cor = signal.correlate2d (Image1, Image2)
    return cor


def rgb2gray(rgb,biases = [1/3,1/3,1/3]):
    """
    Calculate gray image value from RGB value. May include bias values to correct for luminance differences in layers.

    Parameters
    ----------
    rgb : TYPE
        DESCRIPTION.
    biases : TYPE, optional
        DESCRIPTION. The default is [1/3,1/3,1/3].

    Returns
    -------
    gray : numpy.ndarray
        Gray image (2D).

    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #biases = [0.2989, 0.5870, 0.1140]
    gray =  biases[0] * r + biases[1] * g + biases[2] * b
    return gray

def RandImage(X,Y,bindepth = 8):
    """
    Generate a 2D numpy ndarray of dimension 1 and 2 corresponding to X and Y size of the image.
    The values of the pixels in that random array ranges from 0 to the max value of the desired bit depth. (eg. bindepth = 8 : maxvalue = 256 etc.)

    Parameters
    ----------
    X : int
        1st dimension size.
    Y : int
        2nd dimension size.
    bindepth : int, optional
        Bin depth constraining the pixels max value (eg : 8bit = maxvalue 256). The default is 8.
        Please note thtat the minimum value is 0 whatever the bindepth value is.

    Returns
    -------
    numpy.ndarray (2D).

    """
    maxval = max_value_bits(bindepth)
    return np.random.rand(X,Y)*maxval

def RandVideo(X,Y,Time,bindepth = 8):
    """

    Generate a 3D numpy ndarray of dimension 1 and 2 corresponding to X and Y size of the image, and 3rd dimension as time (number of frames).
    The values of the pixels in that random array ranges from 0 to the max value of the desired bit depth. (eg. bindepth = 8 : maxvalue = 256 etc.)

    Parameters
    ----------
    X : int
        1st dimension size
    Y : int
        2nd dimension size.
    Time : 3nd dimension size.
        DESCRIPTION.
    bindepth : int, optional
        Bin depth constraining the pixels max value (eg : 8bit = maxvalue 256). The default is 8.
        Please note thtat the minimum value is 0 whatever the bindepth value is.

    Returns
    -------
    numpy.ndarray (3D) with time as 3rd dimension

    """
    maxval = max_value_bits(bindepth)
    return np.random.rand(X,Y,Time)*maxval

def IM_easypad(binimg,value,**kwargs):
    """
    Pad an image with black (0) borders of a given width (in pixels).
    The padding is homogeneous on the 4 sides of the image.

    Parameters
    ----------
    binimg : numpy.ndarra y(2D)
        Input image.
    value : int
        Pad width (in pixels).
    **kwargs : TYPE
       - mode : "constant" default
       -constant_value : value of pixel if mode is constant

    Returns
    -------
    binimg : numpy.ndarray
        Output image.

    """
    import numpy as np

    mode = kwargs.pop('mode','constant')
    if len(binimg.shape) == 2 :
        binimg = np.pad(binimg, ((value,value),(value,value)), mode, **kwargs )
    elif len(binimg.shape) == 3 :
        binimg = np.pad(binimg, ((value,value),(value,value),(0, 0)), mode, **kwargs )
    else :
        binimg = np.pad(binimg, ((value,value),(value,value),(0, 0),(0, 0)), mode, **kwargs )

    return binimg

#TODO : MAKE SURE IT WORKS (actually just Copy pasta from Shape matching Tracking code)
def ExtractContour(binimg) :
    binimg = Utils.utimg.IM_easypad(binimg, 10 ,mode = 'constant', constant_values = True )

    if binimg.dtype == np.bool :
        binimg = binimg.astype(np.uint8)
        binimg[binimg > 0] = 255

    _ , TempContours, _ = cv2.findContours(binimg, 2, 1)
    return  TempContours[0]

def IM_binarize(image,threshold,**kwargs):
    """
    Binarize an image at a given threshold.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    threshold : int
        Pixel value at which all pixels above are defined as white (255) and all pixels below are defined as black(0).
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    binimg : TYPE
        DESCRIPTION.

    """
    claheobj = kwargs.get("clahe", None)
    if claheobj == "built-in" :
        claheobj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))


    if len(np.shape(image)) > 2 :
        if np.shape(image)[2] < 4:
            image = rgb2gray(image)
        else :
            binimg = np.empty_like(image,dtype = np.bool)
            for Imgindx in range(np.shape(image)[2]):
                if claheobj is not None :
                    tempImg = claheobj.apply(image[:,:,Imgindx])
                else :
                    tempImg = image[:,:,Imgindx]
                _, binimg[:,:,Imgindx] = cv2.threshold(tempImg,threshold,255,cv2.THRESH_BINARY)

            return binimg

    if claheobj is not None :
        tempImg = claheobj.apply(image)
    else :
        tempImg = image
    _, binimg = cv2.threshold(tempImg,threshold,255,cv2.THRESH_BINARY)

    if kwargs.get("bool",True):
        return binimg.astype(np.bool)
    return binimg

def IM_blur(frame,value):
    """
    Blur a 2D image (apply a gaussian 2D filter on it).

    Parameters
    ----------
    frame : numpy.ndarray (2D)
        Input image.
    value : int
        Width of the 2D gaussian curve that is used to look for adjacent pixels values during blurring.

    Returns
    -------
    frame : numpy.ndarray (2D)
        Output image (blurred).

    """
    frame = filters.gaussian(frame, sigma=(value, value), truncate = 6, preserve_range = True).astype('uint8')
    return frame

def GetVideoLen(path):
    """
    Get length (in frame number) of a video file.

    Parameters
    ----------
    path : str
        Path to the video file.

    Raises
    ------
    FileNotFoundError
        If the path to the file is not found.

    Returns
    -------
    length : int
        Number of frames inside the video file.

    """
    if os.path.isfile(path):
        Handlevid = cv2.VideoCapture( path ,cv2.IMREAD_GRAYSCALE )
    else :
        raise FileNotFoundError
    length = int(Handlevid.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

def WriteImageSeq(array, output_folder, **kwargs):
    """
    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    output_folder : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if not os.path.exists(output_folder):
        try :
            os.makedirs(os.path.dirname(output_folder))
        except FileExistsError:
            pass

    if "alerts" in kwargs:
        alerts = kwargs.get("alerts")
    else:
        alerts = False



    if "name" in kwargs:
        output_name = kwargs.get("name")
        output_folder = os.path.join(output_folder,output_name )

    if "extension" in kwargs:
        extension = kwargs.get("extension")
    else:
        if alerts :
            print("Using default extension (.avi) as none was specified")
        extension = ".avi"

    if "fps" in kwargs:
        fps = kwargs.get("fps")
    else:
        fps = 30
        if alerts :
            print("Using default framerate (30 fps) as none was specified")

    if "codec" in kwargs:
        codec = kwargs.get("codec")
    else:
        codec = "MJPG"
        if alerts :
            print("Using default codec (MJPG) as none was specified")

    dtype = kwargs.get("dtype", 'uint8')

    FullOutputPathname = os.path.splitext(output_folder)[0] + extension

    size = np.shape(array)[1], np.shape(array)[0]

    if np.shape(array)[2] > 75 :
        bar = pyprind.ProgBar(np.shape(array)[2],bar_char='░', title=f'Writing video at {FullOutputPathname}')
    else :
        bar = None
        print(f"Writing video at {FullOutputPathname}")

    fourcc = VideoWriter_fourcc(*codec)

    vid = VideoWriter(FullOutputPathname, fourcc, fps, size, True)

    for ImageIndex in range(np.shape(array)[2]):
        if bar is not None :
            bar.update()
        frame = array[:,:,ImageIndex].astype(dtype)
        vid.write(np.repeat(frame[:,:,np.newaxis],3,axis = 2))

    vid.release()

    if os.path.isfile(FullOutputPathname):
        print(f"Success")
    else :
        print(f"File did not create, error, check folder")

def GetImage(path,**kwargs):
    """
    Parameters
    ----------
    path : str
        path to video file.
    **kwargs :
        pos : int. Default : 0
            start frame selected to be returned (0 = start of the video).
        span : int. Default : 1
            number of consecutive frames selected from first frame.
            Will throw error if max number of frames is inferior to pos + span.
        all : bool. Default : False
            if true, pos and span are bypassed and all the video frames will be returned in the numpy array.
        rot : int. Default 0
            0 to 4. Rotates the frames returned by 90° anticlockwise times the amount specified. 0 give sno rotation, 2 gives 180° rotation.

    Returns
    -------
    Output : numpy.ndarray
        Selected frame(s) in a numpy array

    """

    if "rot" in kwargs:
        rotation = kwargs.get("rot")
    else :
        rotation = 0

    if os.path.isfile(path):
        Handlevid = cv2.VideoCapture( path ,cv2.IMREAD_GRAYSCALE )
    else :
        raise ValueError(f"Video file {path} do not exist")

    #width  = int(Handlevid.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(Handlevid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(Handlevid.get(cv2.CAP_PROP_FRAME_COUNT))

    pos = kwargs.get("pos", 0)
    span = kwargs.get('span',1)

    if kwargs.get("all",False):
        pos = 0
        span = length

    if pos < 0 or type(pos) is not int :
        raise ValueError(f"frame start position must be an integer equal or greater than 0, and appears to be {pos}")

    if pos <= length :
        Handlevid.set(cv2.CAP_PROP_POS_FRAMES, pos)
    else :
        raise ValueError(f"Cannot set frame position {pos} greater than video number of frames :{length}")

    if pos + span > length :
        raise ValueError(f"Cannot set combination of pos {pos} and span {span}, would result in last frame {pos+span} greater than video number of frames :{length}")

    if span < 1 or type(span) is not int:
        raise ValueError(f"span must be an integer strictly greater than 0, and appears to be {span}")

    if span > 75 :

        bar = pyprind.ProgBar(span,bar_char='░', title=f'loading video from: {path}')
    else :
        bar = None

    for idx in range(0,span):
        if bar is not None :
            bar.update()
        if idx == 0 :
            _ , TempIMG = Handlevid.read()
            if span > 1 :
                Output = np.empty((np.shape(TempIMG)[0],np.shape(TempIMG)[1],span),dtype = np.uint8)
                Output[:,:,0] = TempIMG[:,:,0]

            else :
                Output = TempIMG[:,:,0]

        else :
            _ , TempIMG = Handlevid.read()
            Output[:,:,idx] = TempIMG[:,:,0]

    if rotation != 0 :
        Output = np.rot90(Output,rotation,axes = (0,1))

    return Output

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

def QuickHist(image, **kwargs):
    """
    Generate histograms of pixel values of an image (color or gray).

    Parameters
    ----------
    image (numpy.ndarray):
        2D or 3D np array with x,y as first two dimensions, and third dimension as RGB channels if color image.

    **color (bool): default = False
        Informs if the array is colored RBG layered data that needs to be averaged as gray to generate histogram
        True : image should be a 3D numpy ndarray (eg.color image)
        False : image should be a 2D numpy ndarray (eg.gray image)

    **bindepth (int): default = 8
        Maximum value of a pixel (8,12,16 bit depths for example)

    **display (bool): default = True
        True to have the function generate a plot.
        False to have the function return a numpy 1D array containing the histogram values.

    Returns
    -------
    None, or numpy.ndarray
        If display is set to False, the function returns a numpy 1D array containing the histogram values..

    """
    import matplotlib.pyplot as plt
    import cv2

    # calculate mean value from RGB channels if presents and flatten to 1D array
    color = kwargs.get("color", False)
    if color :
        vals = image.mean(axis=2).flatten()
    else :
        vals = image.flatten()

    bindepth = kwargs.get("bindepth", 8)
    maxval = max_value_bits(bindepth)

    display = kwargs.get("display", True)
    if display :
        # plot histogram with 255 bins
        b, bins, patches = plt.hist(vals, maxval)
        plt.xlim([0,maxval])
        plt.show()
    else :
        return np.histogram(vals,range = (0,maxval))

class QuickVizGuiDialog(QDialog):
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

        super(QuickVizGuiDialog, self).__init__(parent)

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
        self.VIDwidget.canvas.update_figure( IM_binarize ( self.VIDwidget.RawDATA[:,:,self.VIDwidget.frame  ] , self.BinarizeSlider.value()  ) , **kwargs)
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
    dlg = QuickVizGuiDialog(VideoArray,**kwargs)
    Value = dlg.exec_()
    coords = dlg.Save()
    del qApp, dlg
    if coords is not None :
        return coords
    else :
        return Value

def create_tiff_sequence(array , dirname , sequence_name = "tiff_seq" ):

    from libtiff import TIFF
    digitcount = len(str(np.shape(array)[2]))
    
    for idx in range(np.shape(array)[2]):

        if not os.path.isdir(os.path.join(dirname,sequence_name)) :
             os.makedirs(os.path.join(dirname,sequence_name))

        tiff = TIFF.open(os.path.join(dirname,sequence_name,  "frame_" + str(idx).zfill(digitcount) + '.tiff'), mode='w')
        tiff.write_image(array[:,:,idx])
        tiff.close()


if __name__ == "__main__":

    img = (np.random.rand(20,20,20)*254).astype(np.uint8)
    returnval = VideoDialog(img,mode = "coords")
    print(returnval)