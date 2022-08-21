import math
import scipy
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'supervised_training_iter',
]

# ###############################
# 工具类
# ###############################
class GaussianBlurLayer(nn.Module):
    def __init__(self, channels, kernel_size):
        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)),
            nn.Conv2d(channels, channels, self.kernel_size,
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))


# ###############################
# 训练功能模块
# ###############################
blurer = GaussianBlurLayer(1, 3)
if torch.cuda.is_available():
    blurer.cuda()

def supervised_training_iter(
    omnet, optimizer, image, trimap, gt_matte,
    semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0):
    # gt_matte {Tensor:(12,3,512,512)}
    # image {Tensor:(12,3,512,512)}
    global blurer

    omnet.train()
    optimizer.zero_grad()

    pred_semantic, pred_detail, pred_matte = omnet(image, False)
    boundaries = (trimap < 0.5) + (trimap > 0.5)

    # 语义分割部分的loss
    # Cross Entropy Loss
    gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 16, mode='bilinear')
    gt_semantic = blurer(gt_semantic)
    semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
    semantic_loss = semantic_scale * semantic_loss

    # 细节抠图部分的loss
    # l1 SSIM
    pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
    gt_detail = torch.where(boundaries, trimap, gt_matte)
    detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
    detail_loss = detail_scale * detail_loss

    # 融合部分的loss
    # l1
    pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
    matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)
    matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
                               + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)
    matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
    matte_loss = matte_scale * matte_loss

    #最终损失
    loss = semantic_loss + detail_loss + matte_loss
    loss.backward()
    optimizer.step()

    return semantic_loss, detail_loss, matte_loss


# ###############################
# 主程序
# ###############################

if __name__ == '__main__':
    from matting_dataset import MattingDataset, Rescale, ToTensor, Normalize, ToTrainArray, ConvertImageDtype, GenTrimap
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from models.myNet import OMNet
    from setting import BS, LR, EPOCHS, SEMANTIC_SCALE, DETAIL_SCALE, MATTE_SCALE, SAVE_EPOCH_STEP

    # 图像变形
    transform = transforms.Compose([
        Rescale(512),
        GenTrimap(),
        ToTensor(),
        ConvertImageDtype(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTrainArray()
    ])
    # 加载数据集
    mattingDataset = MattingDataset(transform=transform)

    omnet = torch.nn.DataParallel(OMNet())
    if torch.cuda.is_available():
        omnet = omnet.cuda()
    # 调整优化器SGD
    optimizer = torch.optim.SGD(omnet.parameters(), lr=LR, momentum=0.9)
    # 调整学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * EPOCHS), gamma=0.1)
    dataloader = DataLoader(mattingDataset,
                            batch_size=BS,
                            shuffle=False)

    # 每次epoch
    for epoch in range(0, EPOCHS):
        print(f'epoch: {epoch}/{EPOCHS - 1}')
        for idx, (image, trimap, gt_matte) in enumerate(dataloader):
            semantic_loss, detail_loss, matte_loss = \
                supervised_training_iter(omnet, optimizer, image, trimap, gt_matte,
                                         semantic_scale=SEMANTIC_SCALE,
                                         detail_scale=DETAIL_SCALE,
                                         matte_scale=MATTE_SCALE)
            print(f'{(idx + 1) * BS}/{len(mattingDataset)} --- '
                  f'semantic_loss: {semantic_loss:f}, detail_loss: {detail_loss:f}, matte_loss: {matte_loss:f}\r',
                  end='')
        lr_scheduler.step()
        # 保存中间训练结果

        # 仅保存模型权重参数
    torch.save(omnet.state_dict(), f'omnet_custom_portrait_matting_last_epoch_weight.ckpt')