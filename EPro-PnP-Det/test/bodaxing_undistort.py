import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from scipy import misc
from PIL import Image
import os
import cv2
import imageio

parameter_mapping = {
    # 内参
    'internal_reference': [[1376.20677, 0.000000, 1494.02786], [0.000000, 1384.15178, 889.47556],
                           [0.000000, 0.000000, 1.000000]],
    # 畸变
    'distortion': [-0.245851, 0.061319, 0.002562, -0.005112, 0.000000]
}

base_path = "/home/hongfeng/code_data/BoDaXing/2022-01-19-14-41-27/"
origin_path = base_path + "OriginImg_/"
file_names = os.listdir(origin_path)
file_len = len(file_names)

with imageio.get_writer(uri='result.gif', mode='I', fps=10) as writer:
    for num in range(file_len):
        print("now img idx: ", num)
        file_name = str(num).rjust(6, "0") + ".jpg"
        img_path = origin_path + "2022-01-19-14-41-27_"+ file_name
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        dst = cv2.undistort(img, np.array(parameter_mapping["internal_reference"]), np.array(parameter_mapping["distortion"]), None, None)
        cv2.imencode('.jpg', dst)[1].tofile('result_1.png')

        img_dis = imageio.imread('result_1.png')
        img_ori = imageio.imread(img_path)
        image = cv2.hconcat([img_ori,img_dis])
        writer.append_data(image)

