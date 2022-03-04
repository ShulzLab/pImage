# -*- coding: utf-8 -*-


import os
import pyprind

try :
    import cv2
except ImportError as e :
    cv2 = e


class DefaultReader:
    def __init__(self):
        self.cursor = 0
        
    def _check_edge_cursors(self,down,up):
        if down < 0 :
             down = 0
        if up > self.frames_number :
            up = self.frames_number
        self.set_cursor(down)
        return down, up
    
    def set_cursor(self,value):
        self.cursor = value
    
    def frames(self):
        self.open()
        self.set_cursor(0)
        try :
            while True :
                yield self._get_frame()
        except IOError : 
            return
        #yield from self.frames_span(0,self.frames_number)
        
    def frames_span(self,down,up):
        down, up = self._check_edge_cursors(down,up)
        if up-down > 100 : 
            bar = pyprind.ProgBar(up-down)
            prog = True
        else :
            prog = False
        for i in range(down,up):
            if prog :
                bar.update()
            yield self.frame(i)
            
    def frame(self,frame_id = None):
        if frame_id is None :
            frame_id = self.cursor
        self.cursor += 1
        return self._get_frame(frame_id)
    
            
class AviReader(DefaultReader):
    def __init__(self,file_path):
        super().__init__()
        self.path = file_path

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        #Exception handling here
        self.close()

    def open(self):
        self.file_handle = cv2.VideoCapture( self.path ,cv2.IMREAD_GRAYSCALE )
        #width  = int(Handlevid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(Handlevid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def close(self):
        pass

    @property
    def frames_number(self):
        try :
            #try with ffmpeg if imported, as it showed more accurate results that this shitty nonsense way of calulating frame count of opencv
            import ffmpeg
            return int(ffmpeg.probe(self.path)["streams"][0]["nb_frames"])
        except ImportError :
            return int(self.file_handle.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def set_cursor(self,value):
        self.cursor = value
        self.file_handle.set(cv2.CAP_PROP_POS_FRAMES, self.cursor)

    def _get_frame(self, frame_id=None):
        if frame_id is not None :
            self.file_handle.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        _bool , temp_frame = self.file_handle.read()
        if not _bool:
            raise IOError("end of video file")
        return temp_frame[:,:,0]
        
    
        