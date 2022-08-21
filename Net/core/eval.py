import numpy as np
from glob import glob

from models.myNet import OMNet
from PIL import Image
from predict import predit_matte
import torch.nn as nn
import torch


def cal_mad(pred, gt):
    diff = pred - gt
    diff = np.abs(diff)
    mad = np.mean(diff)
    return mad


def cal_mse(pred, gt):
    diff = pred - gt
    diff = diff ** 2
    mse = np.mean(diff)
    return mse


def load_eval_dataset(dataset_root_dir='/home/zp/JJL/matting/newNet/core/datasets/PPM-100'):
    image_path = dataset_root_dir + '/val/fg/*'
    matte_path = dataset_root_dir + '/val/alpha/*'
    image_file_name_list = glob(image_path)
    image_file_name_list = sorted(image_file_name_list)
    matte_file_name_list = glob(matte_path)
    matte_file_name_list = sorted(matte_file_name_list)

    return image_file_name_list, matte_file_name_list


def eval(modnet: OMNet, dataset):
    mse = total_mse = 0.0
    mad = total_mad = 0.0
    cnt = 0

    for im_pth, mt_pth in zip(dataset[0], dataset[1]):
        im = Image.open(im_pth)
        pd_matte = predit_matte(modnet, im)

        gt_matte = Image.open(mt_pth)
        gt_matte = np.asarray(gt_matte) / 255

        total_mse += cal_mse(pd_matte, gt_matte)
        total_mad += cal_mad(pd_matte, gt_matte)

        cnt += 1
    if cnt > 0:
        mse = total_mse / cnt
        mad = total_mad / cnt

    return mse, mad


if __name__ == '__main__':
    # create MODNet and load the pre-trained ckpt
    omnet = OMNet(backbone_pretrained=False)
    omnet = nn.DataParallel(omnet)

    ckp_pth = 'omnet_custom_portrait_matting_last_epoch_weight.ckpt'
    if torch.cuda.is_available():
        omnet = omnet.cuda()
        weights = torch.load(ckp_pth)
    else:
        weights = torch.load(ckp_pth, map_location=torch.device('cpu'))
    omnet.load_state_dict(weights)
    dataset = load_eval_dataset('/home/zp/JJL/matting/newNet/core/datasets/PPM-100')
    mse, mad = eval(omnet, dataset)
    print(f'mse: {mse:6f}, mad: {mad:6f}')
