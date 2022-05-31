# pImage
 
Massive warning : 
Do not import when using code that needs tensorflow cuda and cudnn as it causes issues (for unknown reasons): namely : 
`I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 5.96G (6404864512 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
Could not load library cudnn_ops_infer64_8.dll. Error code 1455`
 
Usage : 

**Simple guide for video compression :**

``` python
import pImage

input_video_path = r"foo/myfoovideo.seq"
output_video_path = r"bar/myconvertedfoovideo.avi"

converter = pImage.Standard_Converter(input_video_path, output_video_path)
converter.start()
```
Implement progressbar for files of over 100 frames.
Once finished, the script will display `Done` in console

**Under the hood :**

It automatically selects a reader and a writer depending on the extension of your input and output video pathes, and performs reading and writing on different processes to maximize speed in case reading and writing are done on different hard drives. (for multi-core processors)
With optionnal arguments, one can also make the converter execute any function provided to it inbetween the reader and writer to modify the images as wished. (rotation, scale, LUT ,image enhancement, CLAHE,  etc..)

The writer is a class that will generate the file and work variables only at first writer.write(frame) call. That way, it avoids anticipatory declarations of width, height, data type etc at instanciation, an will still work as long as you keep feeding consistant data to the object.
