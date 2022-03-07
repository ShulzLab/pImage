#%% Definitions
import os, math
import numpy as np
import cv2

import warnings

import winerror
import win32api
import win32job

g_hjob = None

def create_job(job_name='', breakaway='silent'):
    hjob = win32job.CreateJobObject(None, job_name)
    if breakaway:
        info = win32job.QueryInformationJobObject(hjob,
                    win32job.JobObjectExtendedLimitInformation)
        if breakaway == 'silent':
            info['BasicLimitInformation']['LimitFlags'] |= (
                win32job.JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)
        else:
            info['BasicLimitInformation']['LimitFlags'] |= (
                win32job.JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        win32job.SetInformationJobObject(hjob,
            win32job.JobObjectExtendedLimitInformation, info)
    return hjob

def assign_job(hjob):
    global g_hjob
    hprocess = win32api.GetCurrentProcess()
    try:
        win32job.AssignProcessToJobObject(hjob, hprocess)
        g_hjob = hjob
    except win32job.error as e:
        if (e.winerror != winerror.ERROR_ACCESS_DENIED or
            sys.getwindowsversion() >= (6, 2) or
            not win32job.IsProcessInJob(hprocess, None)):
            raise
        warnings.warn('The process is already in a job. Nested jobs are not '
            'supported prior to Windows 8.')

def limit_memory(memory_limit):
    if g_hjob is None:
        return
    info = win32job.QueryInformationJobObject(g_hjob, win32job.JobObjectExtendedLimitInformation)
    info['ProcessMemoryLimit'] = memory_limit
    info['BasicLimitInformation']['LimitFlags'] |= (win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY)
    win32job.SetInformationJobObject(g_hjob, win32job.JobObjectExtendedLimitInformation, info)



class memarray(np.memmap):
    def __new__(cls, input_array,**kwargs):
        import random
        rdir = kwargs.pop("root",os.path.abspath("memaps")) 
        if not os.path.isdir(rdir):
            os.makedirs(rdir)
        while True :   
            filename = os.path.join(rdir,"".join([ chr(65+int(random.random()*26)+int(random.random()+0.5)*32) for _ in range(10)]) + ".mmp")
            if not os.path.isfile(filename):
                break       
                
        memobj = super().__new__(cls, filename, shape = input_array.shape , mode = kwargs.pop("mode","w+"), **kwargs )
        whole_slices = tuple([slice(None)]*len(input_array.shape))
        memobj[whole_slices] = input_array[whole_slices]
        
        return memobj
    
    def close(self):
        try :
            self.flush()
            if self._mmap is not None:
                self._mmap.close()
            os.remove(self.filename)
        except ValueError :
            return

class array_video_color(memarray):
    """
    TODO : take as input argument : 
    - 2D_color (treat input array as 3D colored single frame, so convert from shape (X,X,3) to (X,X,3,T))
    - 2D_bw (treat input aray as 2D BW : so convert from shape (X,X) to (X,X,3,T))
    - None (if 3D array : convert as 4D from (X,X,T) to (X,X,3,T) 
           (if 4D array, return out silently the input np.ndarray)
           
    And todo, actually make this take care of the memmaping instead of wraping it inside the VignetteMaker class
    Because this class knows the time, color, and repetition feature of the data inside it's instances.
    Vignete doesn't have to.
    
    Last todo : To account for vignettes where the video data time are shifted, 
    if index outside bounds, return a flat black image 
    (actually, should this be inside Vignette rather ? If we still want some labels on top of empty data....)
        
    Purpose is to mimic the use of a 3D array seamlessly inside some functions 
    that use them for builind vignettes, for flat data that don't change along time, 
    without having to uselessly multiply data usulessly before it needs to be acessed.
    This, in the end, saves some headaches, because it takes as input a 2D or 3D array, and return
    either an instance of this class or a standard np.ndarray, that can be indexed along 
    3 dimensions in the exat same way.
    """
    
    
    
    def __new__(cls, input_array, max_time = 1, array_type = None, **kwargs):
        """If max_time not specified, this class is only usefull to allow a get of one "time frame" 
        at a time and still be compatible with real 3D arrays iterated the same way, without the need for external compatibility checks.
        """
        
        if array_type is None :
            if len(input_array.shape) == 3:
                obj = np.repeat( input_array[:,:,:,np.newaxis].astype(np.uint8), 3 ,axis = 3)
            elif len(input_array.shape) == 4 and input_array.shape[2] ==  3 :
                obj = input_array.astype(np.uint8)
            else :
                raise ValueError("Array shape not understood. Make sure it matches requirements stated in documentation")
        
        elif array_type == "2D_bw" :
            if len(input_array.shape) == 2 :
                if max_time is None :
                    raise ValueError("max_time cannot be None if self is not a 3D array")
                obj = np.repeat(np.repeat( input_array[:,:,np.newaxis].astype(np.uint8), 3 ,axis = 2)[np.newaxis,:,:,:],max_time,axis = 0)
            elif len(input_array.shape) == 3 :
                obj = np.repeat( input_array[:,:,:,np.newaxis], 3 ,axis = 3, dtype = np.uint8)
            else :
                raise ValueError("Array shape not understood. Make sure it matches requirements stated in documentation")
        
        elif array_type == "2D_color" :
            if len(input_array.shape) == 3 :
                if max_time is None :
                    raise ValueError("max_time cannot be None if self is not a 3D array")
                obj = np.repeat( input_array[np.newaxis,:,:,:].astype(np.uint8), max_time , axis = 0)
            elif len(input_array.shape) == 4 and input_array.shape[2] == 3:
                obj = input_array.astype(np.uint8)
            else :
                raise ValueError("Array shape not understood. Make sure it matches requirements stated in documentation")
        else :
            raise ValueError("Missing arguments")
        
        _local_root = os.path.join(os.path.dirname(os.path.dirname(__file__)),"memaps")
        memobj = super().__new__(cls,obj,dtype = kwargs.pop("dtype",np.uint8),root = kwargs.pop("root",_local_root),**kwargs)
        memobj._array_type = array_type
        memobj._max_time = max_time
        return memobj
    
    
try :
    _pass_memset
except :
    assign_job(create_job())
    limit_memory(10000 * 1024 * 1024) #10GB memory Max
    _pass_memset = None
    
class VignetteBuilder():
        
    def __init__(self,target_aspect_ratio = 16/9,maxdim = 1000) :
        self._memap_arrays = []
        self.time_offsets = []
        self.target_aspect_ratio = target_aspect_ratio
        self.border = None
        self.padding = None
        self.maxwidth = self.maxheight = maxdim
        
    def add_video(self,array,**kwargs):
        self.time_offsets.append( kwargs.pop("time_offset",0))
        self._memap_arrays.append( array_video_color(array,**kwargs) )
        #self._memap_arrays[-1].flush()
        
    def grid_layout(self):
        import math
        video_count = len(self._memap_arrays)
        ratios = []
        for columns in range(1,video_count-1):
            lines = math.ceil(video_count/columns)
            aspectratio = (self._memap_arrays[0].shape[2] * columns ) / (self._memap_arrays[0].shape[1] * lines )
            ratios.append( abs( aspectratio / self.target_aspect_ratio - 1 ) )
        
        self.columns = next(index for index , value in enumerate(ratios) if value == min(*ratios)) + 1 
        self.lines = math.ceil(video_count/self.columns)
        
        self.create_grid_background()
        
    def snappy_layout(self):
        pass
        # TODO : add ability to add videos of different shapes and snap them to a dimension of the previously added frames. In that case ,order of frames addition will matter, and 
        # a metadata specifying the x or y dimension to snap onto will also be necessary, as well as a side (top, left ,rigth , bottom)
        # will be quite a pain to code I expect... Not a primordial feature for now.
        
    def get_frame_location(self,index):
        col = 0
        lin = 0
        for i in range(len(self._memap_arrays)):
            if index == i :
                break
            col = col + 1
            if col > self.columns-1 :
                lin = lin + 1 
                col = 0
        return col, lin
    
    def get_frame_ccordinates(self,index):
        col,lin = self.get_frame_location(index)
        x = self.frames_yorigin + (lin * self.frames_interval) + (lin * self.frameheight )
        y = self.frames_xorigin + (col * self.frames_interval) + (col * self.framewidth )
        return x, y, x + self.frameheight, y + self.framewidth
    
    def get_shape(self,index):
        return self.framewidth, self.frameheight
    
    def add_border(self,width):
        self.border = width
        
    def add_padding(self,thickness):
        self.padding = thickness
        
    def create_grid_background(self):
        real_width = self.columns * self._memap_arrays[0].shape[2]
        real_height = self.lines * self._memap_arrays[0].shape[1]
        print(real_width, real_height) 
        self.frames_xorigin = self.frames_yorigin = self.frames_interval = 0
        if  real_width > self.maxwidth or real_height > self.maxheight :
            if real_width / self.maxwidth > real_height / self.maxheight :
                width = self.maxwidth
                height = math.ceil(real_height / (real_width / self.maxwidth))
            else :
                height = self.maxheight
                width = math.ceil(real_width / (real_height / self.maxheight))
        else :
            width = real_width
            height = real_height
            
        self.framewidth = math.ceil( width / self.columns)
        self.frameheight = math.ceil( height / self.lines)
        
        self.background_width = self.framewidth * self.columns
        self.background_height = self.frameheight * self.lines
        
        if self.padding is not None :
            self.background_height = self.background_height + (self.lines - 1 * self.padding)
            self.background_width = self.background_width + (self.columns - 1 * self.padding)
            self.frames_interval = self.padding
            
        if self.border is not None :
            self.background_height = self.background_height + (2*self.border)
            self.background_width = self.background_width + (2*self.border)
            self.frames_xorigin = self.frames_xorigin + self.border
            self.frames_yorigin = self.frames_yorigin + self.border
            
        self.background = np.zeros((self.background_height,self.background_width,3),dtype = np.uint8)
            
    def get_total_duration(self):
        # TODO : use time offset and duration of videos to get the total duration of the video in frames
        pass
    
    def get_time_offset(self,index):
        # use time offset videos to get the absolute positive offset from 0 (requires get_total_duration to  make the  offset positive)
        offset = self.time_offsets[index] 
    
        # TODO : calculate here the offsets etc
        return offset
    
    def set_layout(self, layout_style ):
        if layout_style == "grid" :
            self.grid_layout()
        elif layout_style == "snappy" :
            pass # TODO : maybne add arguments to set layour here to specify is frame index N should be first used to snap onto frame X and then frame Y onto the NX assembly etc...
        else :
            raise ValueError("Unknown layout style")
    
    def frames(self):
        total_time = self.get_total_duration()
        for time_index in range(total_time):
            yield self.frame(time_index)
        
    def frame(self,index):
        import time
        frame = self.background.copy()
        resize_arrays = []
        for i in range(len( self._memap_arrays )):
            time_offset = self.get_time_offset(i)
            _fullsizevig = self._memap_arrays[i][index+time_offset]
            resize_arrays.append(cv2.resize(_fullsizevig, self.get_shape(i), interpolation = cv2.INTER_AREA))
        
        for i in range(len( resize_arrays )):
            x,y,ex,ey = self.get_frame_ccordinates(i)
            frame[x:ex,y:ey,:] = resize_arrays[i]
        return frame
    
    def close(self):
        for array in self._memap_arrays:
            try :
                array.close()
            except ValueError :
                pass
                
# possibnle optimisations : order of indexes in memaps , the select index (time) should be first maybe for faster access : answer, yes it does
# pillow SIMD resize ? https://github.com/uploadcare/pillow-simd
# and possibli, instead of resizing each image then writing in inside the background, maybe write each inside a full size background 
# and resize once to the desires background final shape... 
        
#%% Main test
if __name__ == "__main__" :
    
    #assign_job(create_job())
    #limit_memory(10000 * 1024 * 1024) #10GB memory Max
    
    import os , sys
    import matplotlib.pyplot as plt
    sys.path.append(r"D:\Tim\Documents\Scripts\__packages__")
    import pGenUtils as guti
    import pLabAna as lana
    import pImage
    
    path = r"\\Xps139370-1-ds\data_tim_2\Timothe\DATA\BehavioralVideos\Whisker_Video\Whisker_Topview\Expect_3_mush\Mouse60\210428_1"
    #videos = guti.re_folder_search(path,r".*.avi")
    videos = range(30)
    vb = VignetteBuilder()
    
    for video in videos :
        vid = np.random.rand(1000,1000)*255
        vb.add_video( vid, max_time = 500, array_type = "2D_bw")
        #vb.add_video( pImage.AutoVideoReader(video).frames() )
    vb.set_layout("grid")

#%% Plot

    #plt.imshow(vb.frame(0))
    #vb.close()