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
except ImportError as e :
    pass
import cv2
import numpy as np
import math
import os,sys

from readers import select_extension_reader

available_transforms = {"rotate","crop","annotate","resize","brightness","contrast","gamma","clahe","clipLimit","tileGridSize"}

def TransformingReader(path,**kwargs):

    selected_reader_class = select_extension_reader(path)

    class TransformingPolymorphicReader(selected_reader_class):
           
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
        if clahe or "clipLimit" in kwargs.keys() or "tileGridSize" in kwargs.keys():
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
                frame = image_contrast_brightness(frame,self.contrast,self.brightness)
            if self.gamma is not None :
                frame = image_gamma(frame,self.gamma,self.inv_gamma)
            if self.annotate_params :
                frame = annotate_image(frame, self.annotate_params["text"], **self.annotate_params["params"])
            
            return frame
            
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

def array_gray_to_color( input_array , **kwargs ):
    """
    

    Args:
        input_array (TYPE): DESCRIPTION.
        **kwargs (TYPE): DESCRIPTION.

    Returns:
        TYPE: DESCRIPTION.
        
    Example :
        plt.imshow(pImage.array_gray_to_color(deltaframes[:,:,0],vmin = -0.005, vmax = 0.01,reverse = True))

    """
    
    if np.issubdtype(input_array.dtype, np.integer) :
        vmin, vmax = np.iinfo( input_array.dtype ).min , np.iinfo( input_array.dtype ).max
    else :
        vmin, vmax = np.finfo( input_array.dtype ).min , np.finfo( input_array.dtype ).max
    vmin, vmax = kwargs.get("vmin",vmin) , kwargs.get("vmax",vmax)

    _temp_array = np.interp(input_array.data, (vmin, vmax), (0, 255)).astype(np.uint8)
    if not kwargs.get("reverse",False) :
        _temp_array = np.invert(_temp_array)
    return cv2.applyColorMap(_temp_array, cv2.COLORMAP_JET)

def sequence_gray_to_color(sequence, **kwargs):
    color_sequence = []
    for i in range(sequence.shape[2]):
        color_sequence.append(array_gray_to_color(sequence[:,:,i], **kwargs))
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

def crop(array,*args,**kwargs):
    values = make_crop_params(*args,**kwargs)
    return array[values[0]:array.shape[0]-values[1],values[2]:array.shape[1]-values[3]]

def image_contrast_brightness(array,alpha,beta):
    #alpha = contrast, beta = brightness
    return (np.clip(( array.astype(np.int16) * alpha ) + beta, a_min = 0, a_max = 255)).astype(np.uint8)

def image_gamma(array,gamma,inv= True):
    return apply_lut(array, gamma_lut(gamma,inv))

def image_curve(array,slope,shift):
    return apply_lut(array, curve_lut(slope,shift))

def image_clahe(array,clahe= None,clipLimit = 8, tileGridSize = (5,5) ):
    if clahe is None :
        clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = tileGridSize)
    return clahe.apply(array)
    
def apply_lut(array, lut):
    return np.take(lut,array)
    
def gamma_lut(gamma,inv_gamma = True):
    if inv_gamma : 
        gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table

def curve_lut(slope=1,shift=0):
    return [constrain(to_uint(math.erf(i*slope))+shift) for i in np.linspace(-1,1,256)]

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
    



if __name__ == "__main__" :
    import matplotlib.pyplot as plt

    test  = TransformingReader("tes.avi",rotate = 1)
    