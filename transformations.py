# -*- coding: utf-8 -*-
"""

Boilerplate:
A one line summary of the module or program, terminated by a period.

Rest of the description. Multiliner

<div id = "exclude_from_mkds">
Excluded doc
</div>

<div id = "content_index">

<div id = "contributors">
Created on Thu Mar 10 21:58:07 2022
@author: Timothe
</div>
"""

try :
    from PIL import Image, ImageDraw, ImageFont
except ImportError :
    pass
import cv2
import numpy as np
import math

from readers import _readers_factory

available_transforms = {"rotate","crop","annotate","resize","brightness","contrast","gamma","clahe","clipLimit","tileGridSize","sharpen"}

def TransformingReader(path,**kwargs):

    selected_reader_class = _readers_factory(path,**kwargs)

    class TransformingPolymorphicReader(selected_reader_class):
           
        callbacks = []
        
        rotation_amount = kwargs.pop("rotate",False)
        annotate_params = kwargs.pop("annotate",False)
        crop_params = kwargs.pop("crop",False)
        resize = kwargs.pop("resize",False)
        brightness = kwargs.pop("brightness",0)
        contrast = kwargs.pop("contrast",1)
        brightness_contrast = False if brightness == 0 and contrast == 1 else True
        gamma = kwargs.pop("gamma",None)
        inv_gamma = kwargs.pop("inv_gamma",True)
        clahe = kwargs.pop("clahe",False)
        sharpen_value = kwargs.pop("sharpen",None)
        #parameters for clahe auto set below if not supplied
        if (clahe and isinstance(clahe,bool)) or "clipLimit" in kwargs.keys() or "tileGridSize" in kwargs.keys():
            clahe = cv2.createCLAHE(kwargs.pop("clipLimit",8),kwargs.pop("tileGridSize",(5,5)))
        if crop_params :
            try :
                crop_params = make_crop_params(**crop_params) 
            except TypeError:
                crop_params = make_crop_params(*crop_params)
                
        def _transform_frame(self,frame):
            if self.crop_params :
                frame = crop(frame,*self.crop_params)
            if self.rotation_amount :
                frame = np.rot90(frame,self.rotation_amount,axes=(0, 1))
            if self.resize :
                frame = cv2.resize(frame, (int(frame.shape[1]*self.resize), int(frame.shape[0]*self.resize)), interpolation = cv2.INTER_AREA)
            if self.clahe :
                try :
                    frame = self.clahe.apply(frame)
                except :
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    lab[:,:,0] = self.clahe.apply(lab[:,:,0])
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            if self.brightness_contrast :
                frame = contrast_brightness(frame,self.contrast,self.brightness)
            if self.gamma is not None :
                frame = gamma(frame,self.gamma,self.inv_gamma)
            if self.annotate_params :
                frame = annotate_image(frame, self.annotate_params["text"], **self.annotate_params["params"])
            if self.sharpen_value is not None :
                frame = sharpen_img(frame, self.sharpen_value)
            
            for callback in self.callbacks :
                frame = callback(frame,self)
            
            return frame
        
        def add_callback(self,function):#you can add your own callbacks to a transformingreader 
        #they shoudl be functions that take a frame as input and give back a frame as output
        #the second argument they take is the reader itself, 
        #so that you can implement your own arguments and values withing the function (attach them to the obj before)
            self.callbacks.append(function)
            
        def frames(self):
            for item in self._get_all():
                yield self._transform_frame(item)
                
        def frame(self,frame_id):
            return self._transform_frame(super().frame(frame_id))
                    
            # @property
            # def shape(self):
            #     _shape = super().shape
            #     if self.rotation_amount % 2 :
            #         return (_shape[1], _shape[0] , _shape[2])
            #     return _shape
                
    return TransformingPolymorphicReader(path,**kwargs)

def rescale_to_8bit( input_array, vmin = None, vmax = None,fullrange = False):
    #try to find vmin vmax from input array dtype
    if fullrange:
        if np.issubdtype(input_array.dtype, np.integer) :
            if vmin is None :
                vmin = np.iinfo( input_array.dtype ).min 
            if vmax is None :
                vmax = np.iinfo( input_array.dtype ).max
        else :
            if vmin is None :
                vmin = np.finfo( input_array.dtype ).min
            if vmax is None :
                vmax = np.finfo( input_array.dtype ).max
    else :
        if vmin is None :
            vmin = input_array.min()
        if vmax is None :
            vmax = input_array.max()
        
    try :        
        return np.interp(input_array.data, (vmin, vmax), (0, 255)).astype(np.uint8)
    except AttributeError: #'memoryview' object has no attribute 'data'
        return np.interp(input_array, (vmin, vmax), (0, 255)).astype(np.uint8)
    

def array_gray_to_color( input_array, vmin = None, vmax = None, fullrange = False, cmap = cv2.COLORMAP_JET , reverse = False, mask_where = None, mask_color = 0 ):
    """
    
    Args:
        input_array (TYPE): DESCRIPTION.
        **kwargs (TYPE): DESCRIPTION.

    Returns:
        TYPE: DESCRIPTION.
        
    Example :
        plt.imshow(pImage.array_gray_to_color(deltaframes[:,:,0],vmin = -0.005, vmax = 0.01,reverse = True))

    """
    
    _temp_array = rescale_to_8bit(input_array.data,vmin,vmax,fullrange)
    if not reverse :
        _temp_array = np.invert(_temp_array)
    
    _temp_array = cv2.applyColorMap(_temp_array, cmap)
    
    if mask_where is not None :
        #border_mask_3D = imarrays.mask_sequence(kwargs.get("mask"),deltaframes.shape[2])
        #deltaframes[border_mask_3D[:,:,0] == 0] = kwargs.get("bg_color",0)
        _temp_array[mask_where] = mask_color
        
    return _temp_array

def sequence_gray_to_color(sequence, vmin = None, vmax = None, fullrange = False, cmap = cv2.COLORMAP_JET , reverse = False , mask_where = None, mask_color = 0 ):
    color_sequence = []
    
    for i in range(sequence.shape[2]):
        if mask_where is not None :
            mask = mask_where[:,:,i]
        else :
            mask = None
        color_sequence.append(array_gray_to_color(sequence[:,:,i], vmin,vmax,fullrange,cmap,reverse, mask, mask_color))
    return np.moveaxis(np.array(color_sequence),0,2)

def annotate_image( input_array, text, **kwargs):
    import itertools
    _temp_image = Image.fromarray(input_array)
    x,y = kwargs.get('x',5), kwargs.get('y',5)
    fontsize = kwargs.get('fontsize',100)
    
    if kwargs.get("shadow_size",False) or kwargs.get("shadow_color",False) :
        shadow_size = kwargs.get("shadow_size",5)
        shadow_font = ImageFont.truetype(f"{kwargs.get('font','arial')}.ttf", fontsize + ( shadow_size*1))
        for i, j in itertools.product((-shadow_size, 0, shadow_size), (-shadow_size, 0, shadow_size)):
            ImageDraw.Draw(_temp_image).text( (x+i, y+j) , text , fill=kwargs.get("shadow_color","white") ,font = shadow_font)
        
    default_font = ImageFont.truetype(f"{kwargs.get('font','arial')}.ttf", fontsize)
    ImageDraw.Draw(_temp_image).text( (x,y) , text , fill=kwargs.get('color','black') ,font = default_font)

    return np.array(_temp_image)

def annotate_sequence( sequence, text, **kwargs ) :
    anno_sequence = []
    for i in range(sequence.shape[2]):
        anno_sequence.append(annotate_image(sequence[:,:,i] , text, **kwargs ))
    return np.moveaxis(np.array(anno_sequence),0,2)
        

def make_crop_params(*args,**kwargs):
    if len(args) == 4:
        return args
    if len(args) == 1 and not "value" in kwargs.keys():
        kwargs.update({"value":args[0]})
    def set_value_if_not_none(sval):
        return kwargs.get(sval) if kwargs.get(sval,None) is not None else kwargs.get("value",None)
    sides = ["top","bottom","left","right"]
    values = []
    for side in sides:
        _val = set_value_if_not_none(side)
        if _val is None :
            raise ValueError(f"Must specify at least a general value if specific {side} argument is missing")
        values.append(_val)
    return values


def binarize(image,threshold,boolean = True):
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

    _, binimg = cv2.threshold(image,threshold,255,cv2.THRESH_BINARY)

    if boolean:
        return binimg.astype(np.bool)
    return binimg

def crop(array,*args,**kwargs):
    values = make_crop_params(*args,**kwargs)
    return array[values[0]:array.shape[0]-values[1],values[2]:array.shape[1]-values[3]]

def contrast_brightness(array,alpha,beta):
    #alpha = contrast, beta = brightness
    return (np.clip(( array.astype(np.int16) * alpha ) + beta, a_min = 0, a_max = 255)).astype(np.uint8)

def gamma(array,gamma,inv= True):
    return apply_lut(array, make_lut_gamma(gamma,inv))

def curve(array,slope,shift):
    return apply_lut(array, make_lut_curve(slope,shift))

def clahe(array,clahe= None,clipLimit = 8, tileGridSize = (5,5) ):
    if clahe is None :
        clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = tileGridSize)
    return clahe.apply(array)
    
def apply_lut(array, lut):
    return np.take(lut,array)
    
def make_lut_gamma(gamma,inv_gamma = True):
    if inv_gamma : 
        gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table

def make_lut_curve(slope=1,shift=0):
    return [constrain(to_uint(math.erf(i*slope))+shift) for i in np.linspace(-1,1,256)]

def sharpen_img(img,amount = 0.7):
    from scipy.ndimage.filters import median_filter
    lap = cv2.Laplacian(median_filter(img, 1),cv2.CV_64F)
    return img - amount*lap

def to_uint(value):#-1 - 1 to 0 - 255
    return int((value + 1) * (255/2))

def to_norm(value):#0-255 to -1 - 1
    return int((value - (255/2))/ (255/2))

def constrain(value, mini = 0 , maxi = 255):
    if mini < value < maxi :
        return value
    if value < mini :
        return mini
    else :
        return maxi
    
def grayscale(image,biases = [1/3,1/3,1/3]):
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
    try :
        gray = np.zeros_like(image[:,:,0])
    except IndexError :
        return image
    for color_dim in range(3):
        gray = gray + (image[:,:,color_dim] * biases[color_dim])
    return gray

def pad(image,value,mode = "constant",**kwargs):
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
    return np.pad(image, ((value,value),(value,value)), mode, **kwargs )

def null_image(shape):
    """
    Generate a black image of a given dimension with a white cross on the middle, to use as "no loaded image" user readout.

    Parameters
    ----------
    shape : (tuple) with :
        X : (int) Shape for first dimension on the generated array (X)
        Y : (int) Shape for second dimension on the generated array (Y).

    Returns
    -------
    img : numpy.ndarray
        Blank image with white cross, of [X,Y] shape.

    """
    from skimage.draw import line_aa
    X,Y = shape
    img = np.zeros((X, Y), dtype=np.uint8)
    rr, cc, val = line_aa(0, 0, X-1, Y-1)
    img[rr, cc] = val * 255
    rr, cc, val = line_aa(0, Y-1, X-1, 0)
    img[rr, cc] = val * 255

    return img

def gaussian(frame,value):
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
    from skimage import filters
    frame = filters.gaussian(frame, sigma=(value, value), truncate = 6, preserve_range = True).astype('uint8')
    return frame

if __name__ == "__main__" :
    
    test  = TransformingReader("tes.avi",rotate = 1)
    