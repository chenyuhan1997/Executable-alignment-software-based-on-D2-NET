import argparse
import cv2
import numpy as np
# import imageio
import imageio.v2 as imageio
import plotmatch
from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage import transform



def match(source1,source2):
            start = time.perf_counter()

            _RESIDUAL_THRESHOLD = 30
            #读取图片
            imgfile2 = source1
            imgfile1 = source2
            #开始计时
            start = time.perf_counter()

            #读取图片
            
            image1 = imageio.imread(imgfile1)
            image2 = imageio.imread(imgfile2)


            #计算当前进行时间
            print('read image time is %6.3f' % (time.perf_counter() - start))

            start0 = time.perf_counter()
            #------------------------------------------------------------------------------------------------------------------------------
            #最难部分
            kps_left, sco_left, des_left = cnn_feature_extract(image1,  nfeatures = -1)
            kps_right, sco_right, des_right = cnn_feature_extract(image2,  nfeatures = -1)
            #------------------------------------------------------------------------------------------------------------------------------



            print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))
            start = time.perf_counter()



            #------------------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------------
            #FLANN算法+RANSAN算法的特征点的匹配
            #Flann特征匹配
            #FLANN匹配算法
            FLANN_INDEX_KDTREE = 1
            #字典树。。。。
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=40)

            #邻域搜索，特征匹配匹配器
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des_left, des_right, k=2)


            goodMatch = []
            locations_1_to_use = []
            locations_2_to_use = []


            # 匹配对筛选
            min_dist = 1000
            max_dist = 0
            disdif_avg = 0
            # 统计平均距离差
            for m, n in matches:
                disdif_avg += n.distance - m.distance
            disdif_avg = disdif_avg / len(matches)

            for m, n in matches:
                #自适应阈值
                if n.distance > m.distance + disdif_avg:
                    goodMatch.append(m)
                    p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
                    p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
                    locations_1_to_use.append([p1.pt[0], p1.pt[1]])
                    locations_2_to_use.append([p2.pt[0], p2.pt[1]])
            #goodMatch = sorted(goodMatch, key=lambda x: x.distance)
            print('match num is %d' % len(goodMatch))
            locations_1_to_use = np.array(locations_1_to_use)
            locations_2_to_use = np.array(locations_2_to_use)

            #RANSAC进行文件匹配
            _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                                    transform.AffineTransform,
                                    min_samples=3,
                                    residual_threshold=_RESIDUAL_THRESHOLD,
                                    max_trials=1000)

            print('Found %d inliers' % sum(inliers))

            inlier_idxs = np.nonzero(inliers)[0]
            #最终匹配结果
            matches = np.column_stack((inlier_idxs, inlier_idxs))
            print('whole time is %6.3f' % (time.perf_counter() - start0))

            # Visualize correspondences, and save to file.
            #1 绘制匹配连线
            # plt.rcParams['savefig.dpi'] = 00 #图片像素
            # plt.rcParams['figure.dpi'] = 100 #分辨率
            # plt.rcParams['figure.figsize'] = (5.0, 1.91) # 设置figure_size尺寸
            plt.margins(0,0)
            _, ax = plt.subplots()
            plotmatch.plot_matches(
                ax,
                image1,
                image2,
                locations_1_to_use,
                locations_2_to_use,
                np.column_stack((inlier_idxs, inlier_idxs)),
                plot_matche_points = False,
                matchline = True,
                matchlinewidth = 0.3)
            ax.axis('off')
            ax.set_title('')
            plt.savefig('matching_result',bbox_inches='tight',pad_inches = 0)
            return 'good'

if __name__ == '__main__':

    source1 = 'IR1.jpg'
    source2 = 'VIS1.jpg'
    match(source1, source2)