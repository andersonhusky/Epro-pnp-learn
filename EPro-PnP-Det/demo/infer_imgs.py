"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import argparse
import numpy as np
import imageio
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Infer from images in a directory')
    parser.add_argument('image_dir', help='directory of input images')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--intrinsic', help='camera intrinsic matrix in .csv format',
                        default='demo/nus_cam_front.csv')
    parser.add_argument(
        '--show-dir', 
        help='directory where visualizations will be saved (default: $IMAGE_DIR/viz)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '--show-score-thr', type=float, default=0.3, help='bbox score threshold for visialization')
    parser.add_argument(
        '--show-views',
        type=str,
        nargs='+',
        help='views to show, e.g., "--show-views 2d 3d bev mc score pts orient" '
             'to fully visulize EProPnPDet')
    args = parser.parse_args()
    return args

def create_gif(show_dir):
    file_names = os.listdir(show_dir)
    file_len = int((len(file_names)-1)/2)
    if(not os.path.exists(show_dir +"gif")):
        os.makedirs(show_dir+"gif")
    
    with imageio.get_writer(uri=show_dir+'gif/3d_bev.gif', mode='I', fps=10) as writer:
        for num in range(file_len):
            print("now img idx: ", num)
            file_name_3d = str(num).rjust(6, "0") + "_3d.jpg"
            file_name_bev = str(num).rjust(6, "0") + "_bev.png"
            img_path_3d = show_dir + file_name_3d
            img_path_bev = show_dir + file_name_bev
            img_3d = imageio.imread(img_path_3d)
            img_bev = imageio.imread(img_path_bev)
            img_bev = cv2.resize(img_bev, (1600, 900))
            image = cv2.hconcat([img_3d,img_bev])
            image = cv2.resize(image, (1600, 450))
            writer.append_data(image)

def main():
    args = parse_args()
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_ids = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    if len(gpu_ids) != 1:
        raise NotImplementedError('multi-gpu inference is not yet supported')

    from mmcv.utils import track_iter_progress
    from mmcv.cnn import fuse_conv_bn
    from epropnp_det.apis import init_detector, inference_detector, show_result

    image_dir = args.image_dir
    assert os.path.isdir(image_dir)
    show_dir = args.show_dir
    if show_dir is None:
        show_dir = os.path.join(image_dir, 'viz')
    os.makedirs(show_dir, exist_ok=True)
    cam_mat = np.loadtxt(args.intrinsic, delimiter=',').astype(np.float32)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    model = fuse_conv_bn(model)
    model.test_cfg['debug'] = args.show_views if args.show_views is not None else []

    img_list = []
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.jpeg', '.png']:
            img_list.append(filename)
    img_list.sort()
    kwargs = dict(views=args.show_views) if args.show_views is not None else dict()

    f_target = open(show_dir+"/epro_3d.txt", 'w')
    f_target.truncate(0)
    num = 0
    for img_filename in track_iter_progress(img_list):
        result, data = inference_detector(
            model, [os.path.join(image_dir, img_filename)], cam_mat)
        show_result(
            model, result, data,
            show=False, out_dir=show_dir, show_score_thr=args.show_score_thr,
            **kwargs)

        bbox_3d_results = result[0]["bbox_3d_results"]
        bbox_3d_results = np.concatenate(bbox_3d_results, axis=0)
        score = bbox_3d_results[:, 7]
        mask = score >= args.show_score_thr
        bbox_3d_results = bbox_3d_results[mask]
        for i, bbox_3d_result_single in enumerate(bbox_3d_results):
            new_line = str(num) + " " + str(i) + " " + str(bbox_3d_result_single[3]) + " " + \
                str(bbox_3d_result_single[4]) + " "+ str(bbox_3d_result_single[5]) + " "+ str(bbox_3d_result_single[6])
            f_target.write(new_line)
            f_target.write("\n")
        num = num + 1
    f_target.close()
    

    
    create_gif(show_dir)
    return


if __name__ == '__main__':
    main()
