#%% Definitions
import os, math
import numpy as np
import cv2

import warnings

import winerror
import win32api
import win32job

import readers

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
except NameError:
    assign_job(create_job())
    limit_memory(10000 * 1024 * 1024) #10GB memory Max
    _pass_memset = None
    
    
class vignette_object():
    
    def __init__(self,_object,parent = None):
        if isinstance(_object,array_video_color):
            self.type = "memap"
        elif isinstance(_object,np.ndarray):
            self.type= "array"
        elif isinstance(_object,readers.DefaultReader):
            self.type = "reader"
        elif isinstance(_object,VignetteBuilder):
            self.type = "builder"
        self.static_mode = False
        self.object = _object
        self.parent = parent
            
    @property
    def shape(self):
        if self.type == "memap" :
             return self.object.shape
        elif self.type == "array" :
            shape =  self.object.shape
            return (shape[2],shape[0],shape[1])
        elif self.type == "reader"  :                
            shape =  self.object.shape
            return (shape[2],shape[0],shape[1])
        elif self.type == "builder" :
            if not self.object._layout_ready:
                self.object.get_background_factory()()
            return (self.object.get_total_duration(),self.object.background_height,self.object.background_width)
        
    def set_static(self,static_mode):
        self.static_mode = static_mode
        
    def get_frame(self,frame_id):
        if self.type == "memap" :
            if 0 > frame_id or frame_id > self.object.shape[0]-1 :
                return np.ones((self.object.shape[1],self.object.shape[2],3),dtype = np.uint8) * self.parent.bg_color
            return self.object[frame_id]
        elif self.type == "array" :
            if self.static_mode :
                return self.object[:,:,0]
            if 0 > frame_id or frame_id >= self.object.shape[2] :
                return np.ones((self.object.shape[0],self.object.shape[1],3),dtype = np.uint8)  * self.parent.bg_color
            return self.object[:,:,frame_id]
        elif self.type == "reader" :
            if 0 > frame_id or frame_id > self.object.frames_number - 1 :
                return np.ones((self.object.width,self.object.height,3),dtype = np.uint8)  * self.parent.bg_color
            #only BW readers for now
            if self.object.color is True :
                return self.object[:,:,frame_id]
            return np.repeat(self.object[:,:,frame_id][:,:,np.newaxis], 3, axis = 2)
        elif self.type == "builder" :
            if 0 > frame_id or frame_id > self.object.get_total_duration() :
                return self.object.get_background() 
            return self.object.frame(frame_id)
            
    def close(self):
        if self.type == "memap" :
            self.object.close()
            
    def __str__(self):
        return type(self).__name__ + " Type :" + self.type
    
    def __repr__(self):
        return vignette_object.__str__(self)


def dummy_patch_processor(vignette_builder,patch):
    return patch
    
class VignetteBuilder():
        
    def __init__(self,target_aspect_ratio = 16/9,max_size = 1000,**kwargs) :
        self.v_objects = []
        self.time_offsets = []
        self.sampling_rate_multipliers = []
        self.post_transforms = []
        self.target_aspect_ratio = target_aspect_ratio
        self.set_max_size(max_size)
        self.border = kwargs.get("border",0)
        self.padding = kwargs.get("padding",0)
        self.bg_color = kwargs.get("bg_color",0)
        self.fit_to(kwargs.get("fit_to",0))
        self.set_first_object(kwargs.get("first_object",0))
        self.resize_algorithm = kwargs.get("resize_algorithm",cv2.INTER_AREA)
        self._layout_ready = False
        self._duration_ready = False

    def set_max_size(self, size):
        self.maxwidth = self.maxheight = size
        self._layout_ready = False

    def add_video(self,_object,**kwargs):
        """
        Add a video to create a mosaic. Order of addition matters for grid layout. It can be tuned for snappy layout 
        (if snappy = only 2 videos, no more, no less. One can create snappy layouts with more than 2 videos by stacking builders, see tutorial)
        
        Args:
            _object (TYPE): input.
              Takes as input :
                - any DefaultReader child classes (works with color or black&white inputs)
                - a numpy array. By default, must be a color array with time dimension.
                   Hence, dimensions must be 4, and in this specific order :
                    0 : x
                    1 : y
                    2 : time
                    3 : color 
                    One can also specify other array types , and provide kwarg ``array_mode`` with a certain flag string.
                    flags can be :
                        - np_bwt : expects blackwhite image with time 
                            3d array, dimensions as follow : 
                            0 : x
                            1 : y
                            2 : time
                        - np_bw : expects blackwhite image with no time
                            2D array, dimensions as follow : 
                            0 : x
                            1 : y
                        - np_col : expects color image with no time
                            3d array, dimensions as follow : 
                            0 : x
                            1 : y
                            2 : color
                            
                        Last two np_bw and np_col will create static images, 
                        that won't change across frames indices are called in the builder.
                - another builder. It will feed it's frames to the child builder only when asked to build them. 
                  Can be stacked as many times as RAM allows.
            **kwargs (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        set_static = False
        array_mode =  kwargs.pop("array_mode",None)
        if array_mode == "np_bwt" :#create color dimension from blackwhite image with time 
            _object = np.repeat( _object[:,:,:,np.newaxis] , 3, axis = 3 )
        if array_mode =="np_bw" : #create time then color dimension from blackwhite image with no time (only 2 frames)
            _object = np.repeat( _object[:,:,np.newaxis] , 2 , axis = 2  )
            _object = np.repeat( _object[:,:,:,np.newaxis] , 3 , axis = 3  )
            set_static = True
        if array_mode =="np_col" : #create time dimension from color image with no time (only 2 frames)
            _object = np.repeat( _object[:,:,np.newaxis,:] , 2, axis = 2 )
            set_static = True
            
        self.sampling_rate_multipliers.append( kwargs.pop("sampling_rate_multiplier",1) )
        self.time_offsets.append( kwargs.pop("time_offset",0))
        self.post_transforms.append(kwargs.pop("transform_func", dummy_patch_processor ))
        memmapping = kwargs.pop("memmap_mode", False)
        if memmapping and isinstance(_object,np.ndarray):
            _object = array_video_color(_object,**kwargs)
        vobj = vignette_object(_object, self)
        if set_static :
            vobj.set_static(True)
        self.v_objects.append(vobj )
        self._layout_ready = False
        self._duration_ready = False
        #self.v_objects[-1].object.flush()
        
    def add_border(self,width):
        self.border = width
        self._layout_ready = False
        
    def add_padding(self,thickness):
        self.padding = thickness
        self._layout_ready = False
        
    def get_layout_factory(self):
        if self.layout == "grid":
            return self._apply_grid_layout
        elif self.layout == "snappy":
            return self._apply_snappy_layout
        else :
            raise ValueError("Unknown layout style")
    
    def set_layout(self, layout_style, *args ):
        self.layout = layout_style
        self.layout_args = args
        self._layout_ready = False

    def _apply_grid_layout(self,*args):
        video_count = len(self.v_objects)
        ratios = []
        for columns in range(1,video_count):
            lines = math.ceil(video_count/columns)
            aspectratio = (self.v_objects[0].shape[2] * columns ) / (self.v_objects[0].shape[1] * lines )
            ratios.append( abs( aspectratio / self.target_aspect_ratio - 1 ) )
        
        try :
            self.columns = next(index for index , value in enumerate(ratios) if value == min(ratios)) + 1 
        except StopIteration :
            raise ValueError("Must have more than one video to make a grid layout. Add more videos")
        self.lines = math.ceil(video_count/self.columns)
       
        
    def _apply_snappy_layout(self,alignment = "hori",*args):
        if len(self.v_objects) > 2:
            raise ValueError("Cannot snap more than two objects with one vignette builder in current developpement state. Simply nest builders to achieve the desired number of snapped images.")
            
        if alignment == "hori" or alignment == "horizontal" :
            self.lines = 1
            self.columns = 2
            
        if alignment == "vert" or alignment == "vertical" :
            self.lines = 2
            self.columns = 1
                   
            
    def fit_to(self,index = 0):
        self._fit_to = index
        self._layout_ready = False
        
    def set_first_object(self,index = 0):
        self._f_o = index #first_object
        self._layout_ready = False 
        
        
    def get_frame_time_index(self,object_index,frame_index):
        #frame index expressed in lowest sampling rate (1)
        sampling_rate_multiplier = self.sampling_rate_multipliers[object_index]
        time_offset = self.get_time_offset(object_index)#return already with a - so set + in equation below
        return int(np.round((frame_index + time_offset) / sampling_rate_multiplier))
        
    def get_frame_location(self,index):
        col = 0
        lin = 0
        irange = range(len(self.v_objects)-1,-1,-1) if self.layout == "snappy" and self._f_o else range(len(self.v_objects))
        for i in irange:
            if index == i :
                break
            col = col + 1
            if col > self.columns-1 :
                lin = lin + 1 
                col = 0
        return col, lin
    
    def get_frame_ccordinates(self,index):
        col,lin = self.get_frame_location(index)
        x = self.frames_xorigin + (lin * self.padding) + (lin * self.frameheight )
        y = self.frames_yorigin + (col * self.padding) + (col * self.framewidth )
        return x, y, x + self.frameheight, y + self.framewidth
    
    def _create_snappy_bg(self):
        # object 1 shape does not change
        # self.frameheight and self.framewidth are destined to object 2
        adjust_index = self._fit_to 
        if self.lines == 1 :
            real_height = self.v_objects[adjust_index].shape[1]
            self.frameheight =  real_height #if real_height <= self.maxheight else self.maxheight
            # TODO : change ability to fix a dimension for snappy later.
            scale_multipler = self.frameheight / self.v_objects[not adjust_index].shape[1] 
            self.framewidth = math.ceil( scale_multipler * self.v_objects[not adjust_index].shape[2])
            
            self.background_height = self.frameheight
            self.background_width = self.framewidth + self.v_objects[adjust_index].shape[2]
            
        elif self.columns == 1 :
            real_width = self.v_objects[adjust_index].shape[2]
            self.framewidth =  real_width# if real_width <= self.maxwidth else self.maxwidth
            # TODO : change ability to fix a dimension for snappy later.
            scale_multipler = self.framewidth / self.v_objects[not adjust_index].shape[2] 
            self.frameheight = math.ceil( scale_multipler * self.v_objects[not adjust_index].shape[1])
        
            self.background_width = self.framewidth
            self.background_height = self.frameheight + self.v_objects[adjust_index].shape[1]
        
        else :
            raise ValueError("At least one dimension must be equal to 1")
        
        self.frames_xorigin = self.frames_yorigin = 0
        self._make_spacings()
        self._layout_ready = True
        
    def _create_grid_bg(self):
        real_width = self.columns * self.v_objects[0].shape[2]
        real_height = self.lines * self.v_objects[0].shape[1]
        print("Real dimensions would be : ",real_width, real_height) 
        self.frames_xorigin = self.frames_yorigin = 0
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
        
        self._make_spacings()
              
        self._layout_ready = True
        
    def _make_spacings(self):
        
        #PADDING
        self.background_height = self.background_height + ((self.lines - 1) * self.padding)
        self.background_width = self.background_width + ((self.columns - 1) * self.padding)
        
        #BORDER
        self.background_height = self.background_height + (2*self.border)
        self.background_width = self.background_width + (2*self.border)
        self.frames_xorigin = self.frames_xorigin + self.border
        self.frames_yorigin = self.frames_yorigin + self.border
        
    def get_background_factory(self):
        self.get_layout_factory()(*self.layout_args)
        
        if self.layout == "snappy" :
            return self._create_snappy_bg
        elif self.layout == "grid" :
            return self._create_grid_bg
        else :
            raise ValueError("layout type not understood")
            
    def get_background(self):
        if not self._layout_ready:
            self.get_background_factory()()
        return np.ones((self.background_height,self.background_width,3),dtype = np.uint8) * self.bg_color
        
    def _calculate_time_offsets(self):
        min_value = min(self.time_offsets)
        self._positive_offsets = [offset + min_value for offset in self.time_offsets]
        videos_ends = [self.v_objects[i].shape[0] + self._positive_offsets[i] for i in range(len(self.v_objects))]
        self._total_duration = max(videos_ends)
    
    def get_total_duration(self):
        if self._duration_ready == False :
            self._calculate_time_offsets()
        return self._total_duration
                
    def get_time_offset(self,object_index):
        if self._duration_ready == False :
            self._calculate_time_offsets()
        return -self._positive_offsets[object_index] 
        
    def frames(self):
        total_time = self.get_total_duration()
        for time_index in range(total_time):
            yield self.frame(time_index)
            
    def _get_object_frame(self,object_index,frame_index):
        return self.v_objects[object_index].get_frame(self.get_frame_time_index(object_index,frame_index))
        
    def _snappy_frame_getter(self,frame,index): 
        f_o = self._f_o #first_object
        x , y = self.frames_xorigin , self.frames_yorigin
        if not self._fit_to == f_o :
            patch = cv2.resize(self._get_object_frame(f_o,index), ( self.framewidth, self.frameheight), interpolation = self.resize_algorithm)    
        else :
            patch =  self._get_object_frame(f_o,index)
        ex,ey = x + patch.shape[0] , y + patch.shape[1]
        frame[x:ex,y:ey,:] = self._process_patch(patch,f_o)
        
        col,lin = self.get_frame_location(not f_o)
        x2 = x + (patch.shape[0]*(lin)) + (lin * self.padding) 
        y2 = y + (patch.shape[1]*(col)) + (col * self.padding) 
    
        if self._fit_to == f_o :
            patch = cv2.resize(self._get_object_frame(not f_o,index), ( self.framewidth, self.frameheight), interpolation = self.resize_algorithm)
        else :
            patch = self._get_object_frame(not f_o,index)
        ex2 , ey2 = x2 + patch.shape[0], y2 + patch.shape[1]
            
        frame[x2:ex2,y2:ey2,:] = self._process_patch(patch,not f_o)
        return frame
    
    def _process_patch(self,patch,index):
         return self.post_transforms[index](self,patch) 
    
    def _grid_frame_getter(self,frame,index):
        resize_arrays = []
        for i in range(len( self.v_objects )):
            _fullsizevig = self._get_object_frame(i,index)
            resize_arrays.append(cv2.resize(_fullsizevig, (self.framewidth, self.frameheight), interpolation = self.resize_algorithm))
        
        for i in range(len( resize_arrays )):
            x,y,ex,ey = self.get_frame_ccordinates(i)
            patch = self._process_patch(resize_arrays[i],i)
            frame[x:ex,y:ey,:] = patch 
        return frame
        
    def frame(self,index):
        background = self.get_background()
        if self.layout == "grid":
            return self._grid_frame_getter(background,index)
                
        if self.layout == "snappy" :
            return self._snappy_frame_getter(background,index)

    def close(self):
        for array in self.v_objects:
            try :
                array.close()
            except ValueError :
                pass
                
# possibnle optimisations : order of indexes in memaps , the select index (time) should be first maybe for faster access : answer, yes it does
# pillow SIMD resize ? https://github.com/uploadcare/pillow-simd
# and possibly, instead of resizing each image then writing in inside the background, maybe write each inside a full size background 
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