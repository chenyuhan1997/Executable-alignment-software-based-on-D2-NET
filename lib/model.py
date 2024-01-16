import torch
import torch.nn as nn
import torch.nn.functional as F
#from Ghost_net import GhostBottleneck

class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        self.model = nn.Sequential(
            #VGG模式
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),



            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),




            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=1),



            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        )





        self.num_channels = 512


        self.use_relu = use_relu


        if use_cuda:
            self.model = self.model.cuda()



    def forward(self, batch):

        output = self.model(batch)


        if self.use_relu:

            output = F.relu(output)

        return output







class D2Net(nn.Module):


    def __init__(self, model_file=None, use_relu=True, use_cuda=True):
        super(D2Net, self).__init__()
        self.dense_feature_extraction = DenseFeatureExtractionModule(
            use_relu=use_relu,
            use_cuda=use_cuda
        )




        self.detection = HardDetectionModule()


        self.localization = HandcraftedLocalizationModule()


        if model_file is not None:

            self.load_state_dict(torch.load(model_file, map_location=torch.device('cuda:0'))['model']) #map_location=torch.device('cpu')


    def forward(self, batch):

        _, _, h, w = batch.size()

        dense_features = self.dense_feature_extraction(batch)

        detections = self.detection(dense_features)

        displacements = self.localization(dense_features)

        return {
            'dense_features': dense_features,
            'detections': detections,
            'displacements': displacements
        }


# D2NET= A+B+C+D
#TADE = AA+D2NET+CC+DD










#提取策略
class HardDetectionModule(nn.Module):


    def __init__(self, edge_threshold = 5):

        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold


        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)


        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)


        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)


    def forward(self, batch):
        #单一图像应该是1*512*39*36
        b, c, h, w = batch.size()
        # print('===================================', batch)
        #print('batch_size First ........', b, c, h ,w)
        device = batch.device

        #求出每行最大值
        #返回值是每一行的最大值N和它所在的位置；【0】，指的是只需要最大值N的矩阵，不需要位置矩阵
        depth_wise_max = torch.max(batch, dim=1)[0]
        #print('deep_wise_max First ........',depth_wise_max.shape)

        is_depth_wise_max = (batch == depth_wise_max)

        #print('deep_wise_max Second ........', depth_wise_max.shape)
        #这里输出的是False
        del depth_wise_max







        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)

        #print('local_max First ........',local_max.shape)
        is_local_max = (batch == local_max)
        del local_max
        #print('bbbbbbbbbbbbbbbbbbbbbbb', batch)


        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        # print('kkkkkkkkkkkkk--------', batch)
        # print('lllllllllllll--------', batch.view(-1, 1, h, w))
        # print('dii, dij, djj ........', dii.shape)
        det = dii * djj - dij * dij
        # print('det .... First ', det)
        tr = dii + djj
        #print('tr .................', tr)
        del dii, dij, djj



        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        #像素检测得分
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        #    print('threshold ......', threshold)
        # print('is_not edge ......', is_not_edge)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        # print('detected ...........', detected)
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected










#
class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)

        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)

        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)

        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):

        b, c, h, w = batch.size()

        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)

        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)

        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)




model = HardDetectionModule()
print(model)
