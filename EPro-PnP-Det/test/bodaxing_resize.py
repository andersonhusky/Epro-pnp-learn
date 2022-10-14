import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from scipy import misc
from PIL import Image
import os
import cv2

base_path = "/home/hongfeng/code_data/BoDaXing/2022-01-19-14-41-27/"
target_path = base_path + "0010"
origin_path = target_path + "_ori/"
tmp_path = origin_path + "tmp.jpg"
if(os.path.exists(tmp_path)):
    os.remove(tmp_path)
file_names = os.listdir(origin_path)
file_len = len(file_names)

parameter_mapping = {
    # 内参
    'internal_reference': [[1376.20677, 0.000000, 1494.02786], [0.000000, 1384.15178, 889.47556],
                           [0.000000, 0.000000, 1.000000]],
    # 畸变
    'distortion': [-0.245851, 0.061319, 0.002562, -0.005112, 0.000000]
}

for num in range(file_len):
    print("now img idx: ", num)
    file_name = str(num).rjust(6, "0") + ".jpg"
    img_path = origin_path + "2022-01-19-14-41-27_"+ file_name
    img_re_path = target_path + "/" + file_name
    img_dis_path = target_path + "_dis/" + file_name
    print(img_path)
    print(img_re_path)
    img = Image.open(img_path)
    img_re = img.resize((1600,900),Image.ANTIALIAS)
    img_re = img_re.convert('RGB')
    img_re.save(img_re_path)

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    dst = cv2.undistort(img, np.array(parameter_mapping["internal_reference"]), np.array(parameter_mapping["distortion"]), None, None)
    cv2.imencode('.jpg', dst)[1].tofile(tmp_path)
    img = Image.open(tmp_path)
    img_re = img.resize((1600,900),Image.ANTIALIAS)
    img_re = img_re.convert('RGB')
    img_re.save(img_dis_path)

# python demo/infer_imgs.py demo/ configs/epropnp_det_basic.py demo/epropnp_det_basic.pth --intrinsic demo/nus_cam_front.csv --show-views 3d bev mc