import os
import cv2
import paddlehub as hub
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np


def show_img(img_path, size=8):
    '''
        文件读取图片显示
    '''
    im = imread(img_path)
    plt.figure(figsize=(size, size))
    plt.axis("off")
    plt.imshow(im)


def img_show_bgr(image, size=8):
    '''
        cv读取的图片显示
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(size, size))
    plt.imshow(image)

    plt.axis("off")
    plt.show()


pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")
result = pose_estimation.keypoint_detection(paths=['body.jpg'], visualization=True, output_dir="work/output_pose/")
print(result)

