# -*- coding: utf-8 -*-


import os
import pyprind

try :
    import cv2
except ImportError as e :
    cv2 = e

        
def select_extension_reader(file_path):
    import hiris
    if os.path.splitext(file_path)[1] == ".seq" :
        if isinstance(hiris, ImportError) :
            raise hiris("hiris.py not available in library folder")
        return hiris.HirisReader
    elif os.path.splitext(file_path)[1] in (".avi",".mp4") :
        if isinstance(cv2, ImportError) :
            raise cv2("OpenCV2 cannot be imported sucessfully of is not installed")
        return AviReader
    else :
        raise NotImplementedError("File extension/CODEC not supported yet")


class AutoVideoReader:
    #do not inherit from this class. It only returns other classes factories. 
    #You should inherit from what it yields instead
    def __new__(cls,path,**kwargs):
        from transformations import TransformingReader, available_transforms
            
        if set(kwargs.keys()).intersection(available_transforms) :
            return TransformingReader(path,**kwargs)
        selected_reader_class = select_extension_reader(path)
        return selected_reader_class(path,**kwargs)

class DefaultReader:
    ############## Methods that needs to be overriden :        
    def __init__(self):
        self.color = False
            
    def _get_frame(self,frame_id):
        raise NotImplementedError
        #make it return the specific frame
        
    def _get_all(self):
        raise NotImplementedError
        #make it a yielder for all frames in object (no need for indexing)
        
    def _get_frames_number(self):
        raise NotImplementedError
        #returns the frames number implementing any calculation needed
        
    ############## Methods to override if relevant :  
    def open(self):
        pass
    
    def close(self):
        pass
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()

    ############## Methods to keep :  
    @property
    def frames_number(self):
        try : 
            self._frames_number
        except AttributeError:
            self._frames_number = self._get_frames_number()
        finally :
            return self._frames_number
        
    def sequence(self,start = None, stop = None):
        if start is None :
            start = 0
        if stop is None : 
            stop = self.frames_number   
            
        if stop-start > 100 : 
            bar = pyprind.ProgBar(stop-start)
            prog = True
        else :
            prog = False
        
        for i in range(start,stop):
            if prog :
                bar.update()
            yield self.frame(i)
        
            
    def frames(self):
        yield from self._get_all()
            
    def frame(self,frame_id):
        if frame_id < 0 :
            raise ValueError("Cannot get negative frame ids")
        if self.frames_number is not None and frame_id > self.frames_number-1:
            raise ValueError("Not enough frames in reader")
        return self._get_frame(frame_id)
    
    def __getitem__(self,index):
        import numpy as np
        try : 
            time_start, time_stop = index[2].start,index[2].stop
        except (TypeError, AttributeError):
            _slice  = slice(index[2],index[2]+1)
            time_start, time_stop = _slice.start, _slice.stop
        return np.squeeze(np.moveaxis(np.array(list(self.sequence(time_start,time_stop))),0,2) [(index[0],index[1])])
    
    @property
    def width(self):
        try :
            self._width
        except AttributeError : 
            shape = self.frame(0).shape
            self._height = shape[1]
            self._width = shape[0]
        finally : 
            return self._width
        
    @property
    def height(self):
        try :
            self._height
        except AttributeError : 
            shape = self.frame(0).shape
            self._height = shape[1]
            self._width = shape[0]
        finally : 
            return self._height
    
    @property
    def shape(self):
        return (self.width, self.height , self.frames_number)
                
class AviReader(DefaultReader):
    #only supports grayscale for now
    
    def __init__(self,file_path,**kwargs):
        super().__init__()
        self.path = file_path 
        self.color = kwargs.get("color",False)

    def open(self):
        try : 
            self.file_handle
        except AttributeError : 
            if self.color :
                self.file_handle = cv2.VideoCapture( self.path)# ,cv2.IMREAD_COLOR )
            else :
                self.file_handle = cv2.VideoCapture( self.path ,cv2.IMREAD_GRAYSCALE )
        finally :
            return self.file_handle
        #width  = int(Handlevid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(Handlevid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def close(self):
        try :
            self.file_handle.release()
        except AttributeError:
            pass

    def _get_frames_number(self):

        try :#try with ffmpeg if imported, as it showed more accurate results 
            #that this shitty nonsense way of calulating frame count of opencv
            import ffmpeg
            return int(ffmpeg.probe(self.path)["streams"][0]["nb_frames"])
        except (ImportError, KeyError) :
            self.open()
            frameno = int(self.file_handle.get(cv2.CAP_PROP_FRAME_COUNT))
            return frameno if frameno > 0 else None

    def _get_frame(self, frame_id):
        self.open()
        self.file_handle.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success , temp_frame = self.file_handle.read()
        if not success:
            raise IOError("end of video file")
        if self.color :
            return cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        return temp_frame[:,:,0]
    
    def _get_all(self):
        self.open()
        self.file_handle.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True :
            success, temp_frame = self.file_handle.read()
            if not success :
                break
            if self.color :
                yield temp_frame
            else :
                yield temp_frame[:,:,0]
                
if __name__ == "__main__" :
    import matplotlib.pyplot as plt

    test  = AutoVideoReader("tes.avi",rotate = 1)