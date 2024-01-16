import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions
import numpy as np
import matplotlib.pyplot as plt
import cv2





#投影金字塔
def process_multiscale(image, model, scales=[.25, 0.50, 1.0]):
    #按照前面的处理逻辑，batch变成了1，每次只处理一张图片，尽量减少了GPU的内存消耗
    b, _, h_init, w_init = image.size()



    #print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
    #print('未处理')
    #print(b, _, h_init, w_init)
    device = image.device
    #print(device)
    #只有处理过batch=1的图片可以进行，否则中断程序
    assert(b == 1)
    #给一个张量【3， 0】,tensor([], size=(3, 0))
    all_keypoints = torch.zeros([3, 0])

    #all_description:tensor([], size=(512, 0))
    #这里生成了512个全0张量
    all_descriptors = torch.zeros([
        model.dense_feature_extraction.num_channels, 0
    ])
    #print(all_descriptors)
    #tensor([])
    all_scores = torch.zeros(0)
    #print(all_scores)
    previous_dense_features = None
    banned = None
    #分出3个尺度
    for idx, scale in enumerate(scales):
        #idx为下标，scale依次采样重建
        #这里是分辨率重建，
        current_image = F.interpolate(
            image, scale_factor=scale,
            mode='bilinear', align_corners=True
        )
        #print(current_image)
        #输出第二次
        _, _, h_level, w_level = current_image.size()
        #print('幻影金字塔处理之后:')
        #print(_, _, h_init, w_init)
        #使用卷积输出512个特征图
        dense_features = model.dense_feature_extraction(current_image)


        del current_image

        _, _, h, w = dense_features.size()
        #---------------------------------不用管---------------------------------------------------------------
        # Sum
        if previous_dense_features is not None:
            #累加特征层重采样
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features
        # ---------------------------------不用管---------------------------------------------------------------



        #print('dense_features..........', dense_features.shape)
        #具体流程是这样的：首先输入符合转换后的图片，然后进行等比例的缩放，缩放后就行特征提取（512层）
        #然后进行检测.通过D2-NET损失检测，得到n维向量的特征描述符d^层
        detections = model.detection(dense_features)
        # print('detection...............', detections.shape)
        # ---------------------------------不用管---------------------------------------------------------------
        if banned is not None:
            #重采样
            banned = F.interpolate(banned.float(), size=[h, w]).bool()
            #张量最小值
            detections = torch.min(detections, ~banned)
            #张量最大值
            banned = torch.max(
                torch.max(detections, dim=1)[0].unsqueeze(1), banned
            )
        # ---------------------------------不用管---------------------------------------------------------------



        else:
            banned = torch.max(detections, dim=1)[0].unsqueeze(1)

        fmap_pos = torch.nonzero(detections[0].cpu()).t()
        #print('fmap_ops........', fmap_pos.shape)
        del detections
        # vis


        fig = plt.figure()

        #plt.subplot(2, 1, 2)
        #plt.imshow(img_out)
        for i in range(25):
            vismap = dense_features[0,i,::,::]
            #
            vismap = vismap.cpu()

            #use sigmod to [0,1]
            vismap= 1.0/(1+np.exp(-1*vismap))

            # to [0,255]
            vismap=np.round(vismap*255)
            vismap=vismap.data.numpy()
            plt.subplot(5, 5, i+1)
            plt.axis('off')
            plt.imshow(vismap)
            filename = 'featuremap/CH%d.jpg'% (i)
            cv2.imwrite(filename, vismap)

        plt.tight_layout()
        fig.show()
        #此时我们已经获得了3个尺度上分别的特征描述
        # 恢复位移.
        #数组操作,模型获取N维向量
        displacements = model.localization(dense_features)[0].cpu()
        #print('displacement............', displacements.shape)

        displacements_i = displacements[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        displacements_j = displacements[
            1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        del displacements


        mask = torch.min(
            torch.abs(displacements_i) < 0.5,
            torch.abs(displacements_j) < 0.5
        )
        fmap_pos = fmap_pos[:, mask].to(device)


        valid_displacements = torch.stack([
            displacements_i[mask],
            displacements_j[mask]
        ], dim=0)
        #print('valid_displacements....', valid_displacements.shape)
        del mask, displacements_i, displacements_j


        fmap_keypoints = fmap_pos[1 :, :].float() + valid_displacements.to(device)
        #print('fmap_keypoints.....', fmap_keypoints.shape)
        del valid_displacements

        try:
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.to(device),
                dense_features[0]
            )
        except EmptyTensorError:
            continue

        fmap_pos = fmap_pos[:, ids].to(device)
        fmap_keypoints = fmap_keypoints[:, ids]
        del ids

        keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
        del fmap_keypoints
        #正规则化
        descriptors = F.normalize(raw_descriptors, dim=0).cpu()
        del raw_descriptors

        keypoints[0, :] *= h_init / h_level
        keypoints[1, :] *= w_init / w_level

        fmap_pos = fmap_pos.cpu()
        #print('fmap_pos............last..', fmap_pos.shape)
        keypoints = keypoints.cpu()

        keypoints = torch.cat([
            keypoints,
            torch.ones([1, keypoints.size(1)]) * 1 / scale,
        ], dim=0)
        #print('keypoints...last....', keypoints.shape)

        scores = dense_features[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ].cpu() / (idx + 1)
        #print('scores..........', scores.shape)
        del fmap_pos

        all_keypoints = torch.cat([all_keypoints, keypoints], dim=1)
        all_descriptors = torch.cat([all_descriptors, descriptors], dim=1)
        all_scores = torch.cat([all_scores, scores], dim=0)
        del keypoints, descriptors

        previous_dense_features = dense_features
        del dense_features
    del previous_dense_features, banned
    #.t()就是转置
    keypoints = all_keypoints.t().numpy()
    del all_keypoints
    scores = all_scores.numpy()
    del all_scores
    descriptors = all_descriptors.t().numpy()
    del all_descriptors

    #print('.........', keypoints.shape, '.........', scores.shape, '..........', descriptors.shape)
    return keypoints, scores, descriptors
