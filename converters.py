# -*- coding: utf-8 -*-

"""Boilerplate:
A one line summary of the module or program, terminated by a period.

Rest of the description. Multiliner

<div id = "exclude_from_mkds">
Excluded doc
</div>

<div id = "content_index">

<div id = "contributors">
Created on Tue Oct 12 17:34:44 2021
@author: Timothe
</div>
"""

from multiprocessing import Pool, Manager
import sys, time
from readers import AutoVideoReader
from writers import AutoVideoWriter

class StandardConverter:

    def __init__(self,input_path,output_path, **kwargs):

        m = Manager()
        self.read_queue = m.Queue()
        #self.transformed_queue = m.Queue()
        self.message_queue = m.Queue()

        self.input_path = input_path
        self.output_path = output_path
        self.reader_kwargs = kwargs.get("reader",dict())
        self.writer_kwargs = kwargs.get("writer",dict())

    def start(self):
        with Pool(processes=2) as pool:

            read_process = pool.apply_async(self.read, (AutoVideoReader,self.read_queue,self.message_queue))
            #transform_process = pool.apply_async(self.transform, (self.kwargs.pop("transform_function",_lambda_transform),self.read_queue,self.transformed_queue,self.message_queue))
            write_process = pool.apply_async(self.write, (AutoVideoWriter,self.read_queue,self.message_queue))

            self.last_update = time.time()
            self.r = self.w = 0
            self.max_i = 1
            while True :
                msg = self.message_queue.get()
                self.msg_parser(msg)
                if msg == "End of write process":
                    break
        print("\n" + "Conversion done")

    def msg_parser(self,message):

        if message in ("r" ,"w"):
            exec(f"self.{message} += 1")
            if time.time() - self.last_update > 1 :
                self.last_update = time.time()
                message = fr"""Reading : {(self.r/self.max_i)*100:.2f} % - Writing : {(self.w/self.max_i)*100:.2f} %"""
                print(message, end = '\r', flush=True)
        elif len(message) >= 7 and message[:7] == "frameno" :
            self.max_i = int(message[7:])
        elif len(message) >= 3 and message[:3] == "End" :
            print("\n" + message, end = '', flush=True)

    def read(self, reader_class, read_queue , message_queue):

        with reader_class(self.input_path, **self.reader_kwargs) as vid_read :
            message_queue.put("frameno"+str(vid_read.frames_number))
            for frame in vid_read.frames():
                read_queue.put(frame)
                message_queue.put("r")
        read_queue.put(None)
        message_queue.put("End of read process")
        sys.stdout.flush()

    # def transform(self, transform_function, read_queue, transformed_queue , message_queue):
    #     while True :
    #         frame = read_queue.get()
    #         if frame is None:
    #             transformed_queue.put(None)
    #             break
    #         message_queue.put("t")
    #         transformed_queue.put(transform_function(frame))
    #     message_queue.put("End of transform process")
    #     sys.stdout.flush()

    def write(self,writer_class, read_queue, message_queue):
        with writer_class(self.output_path, **self.writer_kwargs) as vid_write :
            while True :
                frame = read_queue.get()
                if frame is None:
                    break
                message_queue.put("w")
                vid_write.write(frame)
        message_queue.put("End of write process")
        sys.stdout.flush()

if __name__ == "__main__" :
    test = StandardConverter( r"F:\\Timothe\\DATA\\BehavioralVideos\\Whisker_Video\\Whisker_Topview\\Expect_3_mush\\Mouse63\\210521_VSD1\\Mouse63_2021-05-21T16.44.57.avi" ,  
                             r"C:\Users\Timothe\NasgoyaveOC\Professionnel\TheseUNIC\DevScripts\Python\__packages__\pImage\rotst.avi"
                             ,reader = {"rotate":1})
    test.start()