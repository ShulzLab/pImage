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

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os,sys

from readers import select_extension_reader

available_transforms = {"rotate"}

def TransformingReader(path,**kwargs):

    selected_reader_class = select_extension_reader(path)

    class TransformingPolymorphicReader(selected_reader_class):
           
        if kwargs.get("rotate",None) is not None :
            rotation_amount = kwargs.get("rotate")
            def frames(self):
                for item in self._get_all():
                    yield np.rot90(item,self.rotation_amount)
                    
            def frame(self,frame_id):
                return np.rot90(super().frame(frame_id), self.rotation_amount)
            
            @property
            def shape(self):
                _shape = super().shape
                if self.rotation_amount % 2 :
                    return (_shape[1], _shape[0] , _shape[2])
                return _shape
                
    return TransformingPolymorphicReader(path)

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
    if kwargs.get("reverse",True) :
        _temp_array = np.invert(_temp_array)
    return cv2.applyColorMap(_temp_array, cv2.COLORMAP_JET)


def annotate_image( input_array, text, **kwargs):
    import itertools
    _temp_image = Image.fromarray(input_array)
    x,y = kwargs.get('x',5), kwargs.get('y',5)
    fontsize = kwargs.get('fontsize',100)
    
    if kwargs.get("shadow_size",False) or kwargs.get("shadow_color",False) :
        shadow_size = kwargs.get("shadow_size",5)
        shadow_font = ImageFont.truetype(f"{kwargs.get('font','arial')}.ttf", fontsize + ( shadow_size*1))
        for i, j in itertools.product((-shadow_size, 0, shadow_size), (-shadow_size, shadow_size, shadow_size)):
            ImageDraw.Draw(_temp_image).text( (x+i, y+j) , text , fill=kwargs.get("shadow_color","white") ,font = shadow_font)
        
    default_font = ImageFont.truetype(f"{kwargs.get('font','arial')}.ttf", fontsize)
    ImageDraw.Draw(_temp_image).text( (x,y) , text , fill=kwargs.get('color','black') ,font = default_font)

    return np.array(_temp_image)

if __name__ == "__main__" :
    import matplotlib.pyplot as plt

    test  = TransformingReader("tes.avi",rotate = 1)
    