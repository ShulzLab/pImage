import os
import PIL as pillow

def save_as(image,filepath):
    filepath = os.path.abspath(filepath)
    original = pillow.Image.fromarray(image)
    format = os.path.splitext(filepath)[1][1:]
    original.save(filepath, format=format)