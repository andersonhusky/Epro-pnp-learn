import imageio
import os
import cv2
import numpy as np

which_data = "0002_0.2/"
base_path = "/home/hongfeng/Videos/" + which_data
file_names = os.listdir(base_path)
file_len = int((len(file_names)-1)/2)
print(file_len)

if(not os.path.exists(base_path+"gif")):
     os.makedirs(base_path+"gif")
with imageio.get_writer(uri=base_path+'gif/3d_bev.gif', mode='I', fps=10) as writer:
    for num in range(file_len):
        print("now img idx: ", num)
        file_name_3d = str(num).rjust(6, "0") + "_3d.jpg"
        file_name_bev = str(num).rjust(6, "0") + "_bev.png"
        img_path_3d = base_path + file_name_3d
        img_path_bev = base_path + file_name_bev
        img_3d = imageio.imread(img_path_3d)
        img_bev = imageio.imread(img_path_bev)
        img_bev = cv2.resize(img_bev, (1600, 900))
        image = cv2.hconcat([img_3d,img_bev])
        image = cv2.resize(image, (1600, 450))
        writer.append_data(image)