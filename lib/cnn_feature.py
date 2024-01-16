import torch
from lib.model import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale
import scipy
import scipy.io
import scipy.misc
import numpy as np
import cv2 as cv




use_cuda = torch.cuda.is_available()

#引入一个模型
model = D2Net(
    model_file="models\d2_tf.pth",
    use_relu=True,
    use_cuda=use_cuda
)


#设备
device = torch.device("cuda:0")




multiscale = True
#max_edge = 580
max_edge = 2500
max_sum_edges = 5000





# de-net特征提取
def cnn_feature_extract(image, scales = [.25, 0.50, 1.0], nfeatures = 10000):



#这里是像图像变成符合医学和遥感图像的标准图像的关键步骤
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
    #判断图层是不是双通道（基本不可能）
    if len(image.shape) == 2:
        #如果两通道,则扩充到3通道
        image = image[:, :, np.newaxis]
        #将第二维的复制，这样在扩容后就时3通道了
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image 弃用 scipy.misc.imresize.
    resized_image = image


    #如果单边像素高于500，resize
    if max(resized_image.shape) > max_edge:
        resized_image = scipy.misc.imresize(

            #目标图像
            resized_image,
            #按照比例缩放
            max_edge / max(resized_image.shape)

        ).astype('float')


    #重构长宽比像素，同上
    if sum(resized_image.shape[: 2]) > max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')


    #原图和现图的长宽比
    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    #源码处理使用【0， 1】torch张量处理化
    input_image = preprocess_image(
        resized_image,
        preprocessing="torch"
    )
#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------











    #在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
    #requires_grad: 如果需要为张量计算梯度，则为True，否则为False。我们使用pytorch创建tensor时，可以指定requires_grad为True（默认为False），
    #grad_fn: grad_fn用来记录变量是怎么来的，方便计算梯度，y = x*3,grad_fn记录了y由x计算的过程。
    #grad: 当执行完了backward()之后，通过x.grad查看x的梯度值。
    with torch.no_grad():
        #必须执行的multiscale = True
        if multiscale:
            #process_multiscale是核心部分，即多点检测
            keypoints, scores, descriptors = process_multiscale(

                torch.tensor(
                    #一维增加维度，他妈的想不通
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales

            )
            #同上相同不用看
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales
            )






    # 输入图像关联性
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    if nfeatures != -1:
        #根据scores排序
        scores2 = np.array([scores]).T
        res = np.hstack((scores2, keypoints))
        res = res[np.lexsort(-res[:, ::-1].T)]

        res = np.hstack((res, descriptors))
        #取前几个
        scores = res[0:nfeatures, 0].copy()
        keypoints = res[0:nfeatures, 1:4].copy()
        descriptors = res[0:nfeatures, 4:].copy()
        del res
    return keypoints, scores, descriptors
