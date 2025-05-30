import SimpleITK as sitk
import multiprocessing
import logging
from SimpleITK import Image

class ScanProcessor(object):
    
    def __init__(self, *steps):
        self.steps = steps
        super(ScanProcessor, self).__init__()
        
    
    def __call__(self, inputs):
        for s in self.steps:
            inputs = s(inputs)
        return inputs
   
    

class ReadProcessor(object):
    
    def __call__(self, filename):
        return Image()



class AttributesProcessor(object):
    
    def __call__(self, image):
        return dict()



class TransformProcessor(object):
    
    def __call__(self, image):
        return Image()
    


class WriteProcessor(object):
    
    def __call__(self, image, filename):
        return 