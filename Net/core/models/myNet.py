import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SUPPORTED_BACKBONES

# batch & instance 融合 Normalization
class IBNorm(nn.Module):
    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels
        # BN 标准正态分布
        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        # IN 考察每个像素点的信息
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())
        # concat BN和IN结果
        return torch.cat((bn_x, in_x), 1)

# conv+IBN+relu
class Conv2dIBNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]
        # 添加 IBN归一层
        if with_ibn:
            layers.append(IBNorm(out_channels))
        # 添加 relu激活函数
        if with_relu:
            layers.append(nn.ReLU(inplace=True))
        # 构建容器模块
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# SE 注意力机制模块 学习不同通道特征的重要程度
class SEBlock(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # 两个全连接层
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)

# PPM 金字塔池化模型 融合全局信息
class PPMModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        # 创建1，2，3，6 size的 stages，包括池化运算和卷积层
        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in sizes])
        self.bottleneck = nn.Conv2d(in_channels * (len(sizes) + 1), out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, in_channels, size):
        # nn.AdaptiveAvgPool2d()二维自适应平均池化运算 输出1x1 2x2 3x3 6x6
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        # 卷积层
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

# 编码器
# -- Global Semantic Segmentation Branch GSS-branch全局语义分割分支(主分支) 编码器与解码器部分使用PPM模块，上采样部分每层添加SE模块
# -- Local Detail Matting Branch LDM-branch局部细节抠图分支
# -- Fusion Branch F-branch融合分支
# OMNet (Object Matting Net)

#######################################################################
# GSS-branch
#######################################################################
class GSSBranch(nn.Module):
    def __init__(self, backbone):
        super(GSSBranch, self).__init__()
        # [16, 24, 32, 96, 1280]
        enc_channels = backbone.enc_channels
        self.backbone = backbone
        # SE_Block
        self.SE_Block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        # conv_gss16x in_channel 1280 -> out_channel 96 5x5卷积核 步长1 padding填充2 则输入为1284x1284
        self.conv_gss16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        # conv_gss8x in_channel 96 -> out_channel 32 5x5卷积核 步长1 padding2
        self.conv_gss8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        # conv_gssLast in_channel 32 -> out_channel 1 3x3卷积核 步长2 padding1
        self.conv_gssLast = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)


    def forward(self, img, inference):
        enc_features = self.backbone.forward(img)
        # enc_features[0] 倒残差结构[1,16,1,1] 膨胀系数1 输出通道数16 重复次数1 第一次步长1
        # enc_features[1] 倒残差结构[6,24,2,2] 膨胀系数6 输出通道数24 重复次数2 第一次步长2 第二次步长1
        # enc_features[4] 倒残差结构[6,96,3,1] 膨胀系数6 输出通道数96 重复次数3 第一次步长1
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        # backbone输出结果enc32x通过一次se注意力机制模块
        enc32x = self.SE_Block(enc32x)

        # 两次插值采样算法 放大四倍
        gss16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        gss16x = self.conv_gss16x(gss16x)
        gss8x = F.interpolate(gss16x, scale_factor=2, mode='bilinear', align_corners=False)
        gss8x = self.conv_gss8x(gss8x)
        # ppm
        gss8x = self.PPM_module(gss8x)

        # GSS分支预测值 与真值进行比较
        pred_semantic = None
        if not inference:
            gss = self.conv_gssLast(gss8x)
            pred_semantic = torch.sigmoid(gss)

        # 返回GSS语义分割结果，生成的gss8x用于参加LDM分支， 【enc2x, enc4x】为GCC分支阶段结果，参与LDM分支
        return pred_semantic, gss8x, [enc2x, enc4x]

#######################################################################
# LDM-branch
#######################################################################
class LDMBranch(nn.Module):
    # enc_channels = [16, 24, 32, 96, 1280]
    # ldm_channels 32
    def __init__(self, ldm_channels, enc_channels):
        super(LDMBranch, self).__init__()
        # enc_channels[0] 16 -> ldm_channels 32 调整通道数
        self.ldm_enc2x = Conv2dIBNormRelu(enc_channels[0], ldm_channels, 1, stride=1, padding=0)
        # 拼接img输入3通道一起作为输入
        self.conv_enc2x = Conv2dIBNormRelu(ldm_channels + 3, ldm_channels, 3, stride=2, padding=1)
        # enc_channels[1] 24 -> ldm_channels 调整通道数
        self.ldm_enc4x = Conv2dIBNormRelu(enc_channels[1], ldm_channels, 1, stride=1, padding=0)
        # 调整concat后的通道
        self.conv_enc4x = Conv2dIBNormRelu(2 * ldm_channels, 2 * ldm_channels, 3, stride=1, padding=1)
        # LDM分支的卷积块
        self.conv_ldm4x = nn.Sequential(
            Conv2dIBNormRelu(3 * ldm_channels + 3, 2 * ldm_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * ldm_channels, 2 * ldm_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * ldm_channels, ldm_channels, 3, stride=1, padding=1)
        )
        # PPM_module
        self.PPM_module = PPMModule(32, 32, (1, 3, 5))
        # se
        self.se_ldm4x = SEBlock(32, 32)
        self.conv_ldm2x = nn.Sequential(
            # 64 -> 32
            Conv2dIBNormRelu(2 * ldm_channels, 2 * ldm_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * ldm_channels, ldm_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(ldm_channels, ldm_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(ldm_channels, ldm_channels, 3, stride=1, padding=1),
        )
        # 最后一层卷积
        self.conv_ldmLast = nn.Sequential(
            Conv2dIBNormRelu(ldm_channels + 3, ldm_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(ldm_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    # 输入原图、GSSBranch分支的阶段结果enc2x、enc4x，和最终结果gss8x
    def forward(self, img, enc2x, enc4x, gss8x, inference):

        # 先将img下采样到1/2大小,获得1/2的img2x
        img2x = F.interpolate(img, scale_factor=1/2, mode='bilinear', align_corners=False)
        # 将img下采样到1/4大小,获得1/4的img4x
        img4x = F.interpolate(img, scale_factor=1/4, mode='bilinear', align_corners=False)
        # 调整GSS分支中间输出enc2x
        enc2x = self.ldm_enc2x(enc2x)
        # concat img2x和enc2x 作为LDM分支的输入
        ldm4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))

        # 调整GSS分支中间输出enc4x
        enc4x = self.ldm_enc4x(enc4x)
        # concat 前一步的输入和GSS中间结果enc4x
        ldm4x = self.conv_enc4x(torch.cat((ldm4x, enc4x), dim=1))
        gss8x = self.PPM_module(gss8x)
        # 上采样 将GSS分支结果采样2倍
        gss4x = F.interpolate(gss8x, scale_factor=2, mode='bilinear', align_corners=False)
        # 拼接 ldm4x， gss4x， 和img4x，# 99 -> 32   ldm4x:64 gss4x:32 img4x:3
        ldm4x = self.conv_ldm4x(torch.cat((ldm4x, gss4x, img4x), dim=1))
        # se  test__
        ldm4x = self.se_ldm4x(ldm4x)
        # 上采样
        ldm2x = F.interpolate(ldm4x, scale_factor=2, mode='bilinear', align_corners=False)
        ldm2x = self.conv_ldm2x(torch.cat((ldm2x, enc2x), dim=1))

        # LDM分支预测值 与真值进行比较
        pred_detail = None
        # 推断
        if not inference:
            ldm = F.interpolate(ldm2x, scale_factor=2, mode='bilinear', align_corners=False)
            # img通过skip链接和分支最终结果ldm进行concat后输入最后一层conv
            ldm = self.conv_ldmLast(torch.cat((ldm, img), dim=1))
            pred_detail = torch.sigmoid(ldm)

        # 返回GSS语义分割结果，生成的ldm2x用于融合
        return pred_detail, ldm2x

#######################################################################
# F-branch
#######################################################################
class FBranch(nn.Module):
    def __init__(self, ldm_channels, enc_channels):
        super(FBranch, self).__init__()
        # 对gss分支结果gss8x第一次上采样结果进行卷积操作 输入32通道 输出和ldm2x相同通道数
        self.conv_gss4x = Conv2dIBNormRelu(enc_channels[2], ldm_channels, 5, stride=1, padding=2)
        # 对gss和ldm分支结果concat后调整通道
        self.conv_f2x = Conv2dIBNormRelu(2 * ldm_channels, ldm_channels, 3, stride=1, padding=1)
        # 对上采样后的结果和原图进行卷积块操作
        self.conv_fLast = nn.Sequential(
            Conv2dIBNormRelu(ldm_channels + 3, int(ldm_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(ldm_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    # 输入原图img 全局语义分割结果gss8x 局部细节抠图结果ldm2x
    def forward(self, img, gss8x, ldm2x):
        # 先获取gss8x两次上采样的结果gss2x
        gss4x = F.interpolate(gss8x, scale_factor=2, mode='bilinear', align_corners=False)
        gss4x = self.conv_gss4x(gss4x)
        gss2x = F.interpolate(gss4x, scale_factor=2, mode='bilinear', align_corners=False)

        # 将两个分支的结果concat、卷积调整通道
        f2x = self.conv_f2x(torch.cat((gss2x, ldm2x), dim=1))
        # 融合后的f2x上采样
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)
        # 对上采样后的结果与输入concat后进行卷积块操作
        f = self.conv_fLast(torch.cat((f, img), dim=1))

        # 预测的alpha matte
        pred_matte = torch.sigmoid(f)
        return pred_matte

#######################################################################
# OMNet
#######################################################################
class OMNet(nn.Module):
    # 输入 原图像的3通道in_channels ldm细节抠图分支的输入32通道
    def __init__(self, in_channels=3, ldm_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=True):
        super(OMNet, self).__init__()

        self.in_channels = in_channels
        self.ldm_channels = ldm_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        # backbone部分
        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.gss_branch = GSSBranch(self.backbone)
        self.ldm_branch = LDMBranch(self.ldm_channels, self.backbone.enc_channels)
        self.f_branch = FBranch(self.ldm_channels, self.backbone.enc_channels)

        # self.ppm_module = PPMModule()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()

    def forward(self, img, inference):
        # 获取gss分支的结果
        pred_semantic, gss8x, [enc2x, enc4x] = self.gss_branch(img, inference)
        # 获取ldm分支的结果
        pred_detail, ldm2x = self.ldm_branch(img, enc2x, enc4x, gss8x, inference)
        # 获取预测结果
        pred_matte = self.f_branch(img, gss8x, ldm2x)

        return pred_semantic, pred_detail, pred_matte

    # 训练用的操作
    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    # 卷积初始化
    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu'
        )
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
