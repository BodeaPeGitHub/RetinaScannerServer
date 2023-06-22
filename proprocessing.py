import pandas as pd
import os
import random
import math
import numpy as np

from PIL import Image
from skimage import io, exposure
import cv2
from received_image import ReceivedImage


class PrepocessingImages:
    
    def __init__(self, image_size = (512, 512)):
        self.__image_size = image_size
        pass

    def __variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()
    
    def __resize(self, image):
        return cv2.resize(image, self.__image_size)
    
    def preprocess_image(self, image):
        blur_value = self.__variance_of_laplacian(image)
        return self.__resize(image)
    
    def preprocess_images(self, images: list[ReceivedImage]):
        for image in images:
            image.image = self.__resize(image=image.image)
        return images
        
