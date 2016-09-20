import numpy as np
import cv2

def loadImage(imagepath, mean_image = [103.939, 116.779, 123.68]):
    img = cv2.resize(cv2.imread(imagepath), (224, 224))
    mean_pixel = mean_image # [103.939, 116.779, 123.68]
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)
    return img
