import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image
import os
import cv2

base_path = "/home/hongfeng/code_data/kitti_tracking/data_tracking_image_2/training/image_02/"
whitch_data = "0004/"
origin_path = base_path + whitch_data + "img/"
target_path = base_path + whitch_data + "img_/"
file_names = os.listdir(origin_path)
file_len = len(file_names)

for num in range(file_len):
    print("now img idx: ", num)
    file_name = str(num).rjust(6, "0") + ".png"
    img_path = origin_path + file_name
    img_re_path = target_path + file_name
    print(img_path)
    print(img_re_path)
    img = Image.open(img_path)
    img_re = img.resize((1600,900),Image.ANTIALIAS)
    img_re = img_re.convert('RGB')
    img_re.save(img_re_path)

# python demo/infer_imgs.py demo/ configs/epropnp_det_basic.py demo/epropnp_det_basic.pth --intrinsic demo/nus_cam_front.csv --show-views 3d bev mc