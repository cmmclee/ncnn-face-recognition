import cv2
import facerecognition
import numpy as np

img = cv2.imread("Aaron_Sorkin_0001.bmp")
facerecg = facerecognition.FaceRecognition("./models", 0.70)


image_char = img.astype(np.uint8).tostring()

rets = facerecg.recognize(img.shape[0], img.shape[1], image_char)
print(rets)
