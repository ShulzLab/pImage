# -*- coding: utf-8 -*-

"""Boilerplate:
A one line summary of the module or program, terminated by a period.

Rest of the description. Multiliner

<div id = "exclude_from_mkds">
Excluded doc
</div>

<div id = "content_index">

<div id = "contributors">
Created on Tue Oct 12 18:54:37 2021
@author: Timothe
</div>
"""

import os, warnings
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc


def select_extension_writer(file_path):
    if os.path.splitext(file_path)[1] == ".avi" :
        return AviWriter
    if os.path.splitext(file_path)[1] == ".tiff" :
        return TiffWriter
    else :
        raise NotImplementedError("File extension/CODEC not supported yet")


class AutoVideoWriter:
    
    def __new__(cls,path):
        selected_writer_class = select_extension_writer(path)
        return selected_writer_class(path)

class DefaultWriter:
    
    ############## Methods that needs to be overriden :      
    def __init__(self):
        pass
    
    def _write_frame(self,array):
        raise NotImplementedError
        #implement the actual writing of one frame to the file output
        
        
    ############## Methods to overrride if necessary :  
    def open(self):
        pass
    
    def close(self):
        pass
    
    ############## Methods to keep :      
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, type, value, traceback):
        #Exception handling here
        self.close()
        
    def write(self,array):
        self._write_frame(array)

class AviWriter(DefaultWriter):
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
        
        self.file_handle = None
        
    def _write_frame(self,array):

        if self.file_handle is None :
            self.size = np.shape(array)[1], np.shape(array)[0]
            self.file_handle = VideoWriter(self.path, self.fourcc, self.fps, self.size, True)#color is always True because...

        frame = array.astype(self.dtype)
        if len(frame.shape) < 3 :
            frame = np.repeat(frame[:,:,np.newaxis],3,axis = 2)#...I just duplicate 3 times gray data if it isn't
        elif self.rgbconv is not None :
            frame = eval( f"cv2.cvtColor(frame, cv2.COLOR_{self.rgbconv})" )
        self.file_handle.write(frame)

    def close(self):
        if self.file_handle is None :
            warnings.warn("No data has been given, video was not created")
            return 
        self.file_handle.release()

class TiffWriter(DefaultWriter):
    try :
        from libtiff import TIFF as tiff_writer
    except ImportError as e:
        tiff_writer = e
    def __init__(self,path,**kwargs):
        self.path =  os.path.dirname(path)
        self.file_prefix = os.path.splitext(os.path.basename(path))[0]
        self.index = 0
        
    def _make_full_fullpath(self,index):
        if not os.path.isdir(self.path) :
            os.makedirs(self.path)
        return os.path.join(self.path,self.file_prefix + f"_{str(index).zfill(5)}.tiff")
        
    def _write_frame(self,array):
        _fullpath = self._make_full_fullpath(self.index)
            
        tiff_writer = self.tiff_writer.open(_fullpath, mode = "w")
        tiff_writer.write_image(array)
        tiff_writer.close()
        self.index += 1






