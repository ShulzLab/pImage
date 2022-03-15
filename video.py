# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:30:45 2020

@author: Timothe
"""

import os,sys
import numpy as np
import pyprind
from cv2 import VideoWriter, VideoWriter_fourcc, VideoCapture
import cv2
import warnings

### OLD FILE, PREFER USING CLASSES IN 
### READERS, WRITERS, AND CONVERTERS

####
# def select_reader(file_path):
#     if os.path.splitext(file_path)[1] == ".seq" :
#         if isinstance(hiris, ImportError) :
#             raise hiris("hiris.py not available in library folder")
#         return hiris.HirisReader
#     elif os.path.splitext(file_path)[1] in (".avi",".mp4") :
#         if isinstance(cv2, ImportError) :
#             raise cv2("OpenCV2 cannot be imported sucessfully of is not installed")
#         return readers.AviReader
#     else :
#         raise NotImplementedError("File extension/CODEC not supported yet")


# class AutoVideoReader:
    
#     def __new__(cls,path,**kwargs):
#         selected_reader_class = select_reader(path)
#         return selected_reader_class(path,**kwargs)
    
    # def __init__(self,path, **kwargs):
    #     self.path = path
    #     selected_reader_class = select_reader(path)
    #     self.reader = selected_reader_class(path, **kwargs)

    # def __enter__(self):
    #     self.reader.open()
    #     return self

    # def __exit__(self, type, value, traceback):
    #     self.reader.close()

    # def frames_yielder(self,start=None,stop=None):
    #     """
    #     Use this to get frames one by one in a for loop 
    #     (read at the file on frame per iteration so the ram stays clean)
    #     used in converters to read on the fly on a separate multiprocessing.

    #     Args:
    #         start (TYPE, optional): First frame index. None if start at frame 0 of the file. Defaults to None.
    #         stop (TYPE, optional): Last frame index. None if stops at last frame of the file. Defaults to None.

    #     Yields:
    #         np.array(2D): A frame per iteration in a loop context or a generator.

    #     """
    #     if start is None :
    #         yield from self.reader.frames()
    #     else :
    #         yield from self.reader.frames_span(start,stop)

    # def frames(self,start=None,stop=None):
    #     """
    #     Wrapping a call to frames_yielder and getting all frames at once, storing them inside a numpy array 
    #     (3rd dimension = time)
    #     This uses more Ram but has the benefit of accessing the file only once.

    #     Args:
    #         start (TYPE, optional): First frame index. None if start at frame 0 of the file. Defaults to None.
    #         stop (TYPE, optional): Last frame index. None if stops at last frame of the file. Defaults to None.

    #     Returns:
    #         np.array(3D): A frame array (3d dimension = time).

    #     """
    #     return np.moveaxis(np.array(list(self.frames_yielder(start,stop))),0,2)

    # @property
    # def frames_number(self):
    #     return self.reader.frames_number
    
    # def __getitem__(self,value):
    #     if isinstance(value,(list,tuple)) and len(value) == 2:
    #         return self.frames(*value)
    #     else :
    #         return self.reader.frame(value)
    
    # def frame(self,frame_nb):
    #     return self.reader.frame(frame_nb)
    


class frames_ToAVI:
    def __init__(self,path,**kwargs):
        """
        Creates an object that contains all parameters to create a video,
        as specified with kwargs listed below.
        The first time object.addFrame is called, the video is actually opened,
        and arrays are written to it as frames.
        When the video is written, one can call self.close() to release
        python handle over the file or it is implicity called if used in structure :
        ```with frames_ToAVI(params) as object``` wich by the way is advised for better
        stability.

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            - fps :
                playspeed of video
            - codec :
                4 char string representing an existing CODEC installed on  your machine
                "MJPG" is default and works great for .avi files
                List of FOURCC codecs :
                https://www.fourcc.org/codecs.php
            - dtype :
            - rgbconv :

            root
            filename


        Returns
        -------
        None.

        """

        filename = kwargs.get("filename",None)
        root = kwargs.get("root",None)
        if root is not None :
            path = os.path.join(root,path)
        if filename is not None :
            path = os.path.join(path,filename)
        if not os.path.isdir(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])

        self.path = path

        self.rgbconv = kwargs.get("rgbconv",None)
        # /!\ : if data from matplotlib exported arrays : color layers are not right
        #necessary to add parameter  rgbconv = "RGB2BGR"

        self.fps = kwargs.get("fps", 30)
        self.codec = kwargs.get("codec", "MJPG")
        self.dtype = kwargs.get("dtype", 'uint8')

        self.fourcc = VideoWriter_fourcc(*self.codec)

    def __enter__(self):
        self.vid = None
        return self

    def __exit__(self, type, value, traceback):
        #Exception handling here
        if self.vid is None :
            warnings.warn("No data has been given, video was not created")
            return
        self.close()

    def addFrame(self,array):

        if self.vid is None :
            self.size = np.shape(array)[1], np.shape(array)[0]
            self.vid = VideoWriter(self.path, self.fourcc, self.fps, self.size, True)#color is always True because...

        frame = array.astype(self.dtype)
        if len(frame.shape) < 3 :
            frame = np.repeat(frame[:,:,np.newaxis],3,axis = 2)#...I just duplicate 3 times gray data if it isn't
        elif self.rgbconv is not None :
            frame = eval( f"cv2.cvtColor(frame, cv2.COLOR_{self.rgbconv})" )
        self.vid.write(frame)

    def close(self):
        self.vid.release()




def Array_ToAVI(VideoArray,path,**kwargs): #ENHANCED LISIBILITY, USE CASES, AND REDUCED VERBOSITY

    filename = kwargs.get("filename",None)
    root = kwargs.get("root",None)

    if root is not None :
        path = os.path.join(root,path)
    if filename is not None :
        path = os.path.join(path,filename)

    if not os.path.split(path)[0] == '' and not os.path.isdir(os.path.split(path)[0]) :
        os.makedirs(os.path.split(path)[0])

    fps = kwargs.get("fps", 30)
    codec = kwargs.get("codec", "MJPG")
    dtype = kwargs.get("dtype", 'uint8')
    size = np.shape(VideoArray)[1], np.shape(VideoArray)[0]
    fourcc = VideoWriter_fourcc(*codec)

    vid = VideoWriter(path, fourcc, fps, size, True)

    bar = pyprind.ProgBar(np.shape(VideoArray)[2],bar_char='▓',title=f'\nWriting video at {path}')
    for ImageIndex in range(np.shape(VideoArray)[len(VideoArray.shape)-1]):
        bar.update()
        frame = VideoArray[...,ImageIndex].astype(dtype)
        if len(frame.shape) < 3 :
            frame = np.repeat(frame[:,:,np.newaxis],3,axis = 2)
        vid.write(frame)

    vid.release()
    print(f"Video compression at {path} sucessfull\n")





# def VideoArrayWrite(VideoArray,output_folder,**kwargs):
#     warnings.warn("VideoArrayWrite is deprecated, use Array_ToAVI instead", DeprecationWarning,"-wd")

#     if not os.path.exists(output_folder):
#         try :
#             os.makedirs(output_folder)
#         except FileExistsError:
#             pass

#     if "alerts" in kwargs:
#         alerts = kwargs.get("alerts")
#     else:
#         alerts = True

#     if "output_name" in kwargs:
#         output_name = kwargs.get("output_name")
#     else:
#         #ERROR 2 FLAG: LOOK FOR REASON HERE : BEGINING
#         if "input_path" in kwargs:
#             input_path = kwargs.get("input_path")
#             path,file = os.path.split(input_path)
#             if file.endswith(".seq") and path != "":
#                 output_name = os.path.basename(path)
#             else:
#                 if path == "" or path == None:
#                     output_name = file
# #                    sys.exit("ERROR 2 INVALID_PATH : You must either specify a filename with : output_name = ""Nameofyourfile"" (better practice is doing it iteratively) or the path input to get the seq file, used in HirisSeqReader, with input_path = ""pathtoyourvideo"" ")
#                 else :
#                     if "\\" not in path:
#                         output_name = path
#                     else :
#                         output_name = os.path.basename(path)
#         else :
#             #ERROR 2 FLAG : LOOK FOR REASON HERE : END
#             sys.exit("ERROR 2 FILE_NOT_FOUND : You must either specify a filename with : output_name = ""Nameofyourfile"" (better practice is doing it iteratively) or the path input to get the seq file, used in HirisSeqReader, with input_path = ""pathtoyourvideo"" ")
#     print(output_name)
#     if "extension" in kwargs:
#         extension = kwargs.get("extension")
#     else:
#         if alerts :
#             print("Using default extension (.avi) as none was specified")
#         extension = ".avi"

#     if "fps" in kwargs:
#         fps = kwargs.get("fps")
#     else:
#         fps = 30
#         if alerts :
#             print("Using default framerate (30 fps) as none was specified")

#     if "codec" in kwargs:
#         codec = kwargs.get("codec")
#     else:
#         codec = "MJPG"
#         if alerts :
#             print("Using default codec (MJPG) as none was specified")

#     FullOutputPathname = os.path.join(output_folder,output_name+extension)

#     size = np.shape(VideoArray)[1], np.shape(VideoArray)[0]

#     #np.size(VideoArray,1) , np.size(VideoArray,0)

#     bar = pyprind.ProgBar(np.shape(VideoArray)[2],bar_char='▓',title=f'\nWriting video at {FullOutputPathname}')

#     fourcc = VideoWriter_fourcc(*codec)

#     vid = VideoWriter(FullOutputPathname, fourcc, fps, size, True)

#     for ImageIndex in range(np.shape(VideoArray)[2]):

#         bar.update()
#         frame = VideoArray[:,:,ImageIndex].astype('uint8')
#         vid.write(np.repeat(frame[:,:,np.newaxis],3,axis = 2))

#     vid.release()


#     print(f"Video compression at {FullOutputPathname} sucessfull\n")

# def Array_ToAVI(VideoArray,output_folder,**kwargs): #PRIORITIZE THIS FUNCTION  NAME COHERENCE

#     if not os.path.exists(output_folder):
#         try :
#             os.makedirs(output_folder)
#         except FileExistsError:
#             pass

#     if "alerts" in kwargs:
#         alerts = kwargs.get("alerts")
#     else:
#         alerts = True

#     if "output_name" in kwargs:
#         output_name = kwargs.get("output_name")
#     else:
#         #ERROR 2 FLAG: LOOK FOR REASON HERE : BEGINING
#         if "input_path" in kwargs:
#             input_path = kwargs.get("input_path")
#             path,file = os.path.split(input_path)
#             if file.endswith(".seq") and path != "":
#                 output_name = os.path.basename(path)
#             else:
#                 if path == "" or path == None:
#                     output_name = file
# #                    sys.exit("ERROR 2 INVALID_PATH : You must either specify a filename with : output_name = ""Nameofyourfile"" (better practice is doing it iteratively) or the path input to get the seq file, used in HirisSeqReader, with input_path = ""pathtoyourvideo"" ")
#                 else :
#                     if "\\" not in path:
#                         output_name = path
#                     else :
#                         output_name = os.path.basename(path)
#         else :
#             #ERROR 2 FLAG : LOOK FOR REASON HERE : END
#             sys.exit("ERROR 2 FILE_NOT_FOUND : You must either specify a filename with : output_name = ""Nameofyourfile"" (better practice is doing it iteratively) or the path input to get the seq file, used in HirisSeqReader, with input_path = ""pathtoyourvideo"" ")
#     print(output_name)
#     if "extension" in kwargs:
#         extension = kwargs.get("extension")
#     else:
#         if alerts :
#             print("Using default extension (.avi) as none was specified")
#         extension = ".avi"

#     if "fps" in kwargs:
#         fps = kwargs.get("fps")
#     else:
#         fps = 30
#         if alerts :
#             print("Using default framerate (30 fps) as none was specified")

#     if "codec" in kwargs:
#         codec = kwargs.get("codec")
#     else:
#         codec = "MJPG"
#         if alerts :
#             print("Using default codec (MJPG) as none was specified")

#     dtype = kwargs.get("dtype", 'uint8')


#     FullOutputPathname = os.path.join(output_folder,output_name+extension)

#     size = np.shape(VideoArray)[1], np.shape(VideoArray)[0]

#     #np.size(VideoArray,1) , np.size(VideoArray,0)

#     bar = pyprind.ProgBar(np.shape(VideoArray)[2],bar_char='▓',title=f'\nWriting video at {FullOutputPathname}')

#     fourcc = VideoWriter_fourcc(*codec)

#     vid = VideoWriter(FullOutputPathname, fourcc, fps, size, True)

#     for ImageIndex in range(np.shape(VideoArray)[2]):

#         bar.update()
#         frame = VideoArray[:,:,ImageIndex].astype(dtype)
#         vid.write(np.repeat(frame[:,:,np.newaxis],3,axis = 2))

#     vid.release()


#     print(f"Video compression at {FullOutputPathname} sucessfull\n")

def AVI_ToArray(path,**kwargs):

    rotation = kwargs.get("rotation", 0)
    videoHandle = VideoCapture(path,cv2.IMREAD_GRAYSCALE)

    if (videoHandle.isOpened()== False):
        print("Error opening video stream or file")

    width  = int(videoHandle.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoHandle.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(videoHandle.get(cv2.CAP_PROP_FRAME_COUNT))

    bar = pyprind.ProgBar(length/10,bar_char='░', title=f'loading BEHAVIOR :{path}')

    FrameArray = np.empty((height,width,length),dtype=np.uint8)

    for i in range(length):
        _ , IMG = videoHandle.read()

        FrameArray[:,:,i] = IMG[:,:,0]

        if i % 10 == 0:
            bar.update()

    del bar

    if rotation != 0 :
        FrameArray = np.rot90(FrameArray, rotation, axes=(0, 1))

    return FrameArray

if __name__ == "__main__" :
    
    import ffmpeg
    import matplotlib.pyplot as plt
    
    with AutoVideoReader( r"C:\Users\Timothe\Desktop\Testzone\Mouse33_2020-07-06T16.15.31.avi" ) as test :
    #     print(test.reader)
         for frame in test.frames():
             print(frame)
             plt.imshow(frame)
             plt.show()
