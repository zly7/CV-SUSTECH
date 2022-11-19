import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    kernel_size = 3
    how_many_features = len(x)
    cell_length = 4  # 每个特征点描述大小,代表这个特征有16个格子,每个格子8个方向，然后用欧几里得距离
    each_cell_influence_length = feature_width // cell_length  # 每个cell囊括的特征大小
    # 这上面有个巨坑，就是这样会引入浮点数，然后导致数组不能索引
    how_many_orientation = 8
    # print(image.shape)
    # image = np.pad(image, feature_width // 2) # todo：这里pad之后就改变图片大小了，最开始没发现简直在搞笑
    # print(image.shape)
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    magnitudes = np.sqrt(np.square(dx) + np.square(dy))
    # magnitudes = cv2.filter2D(magnitudes, ddepth=-1, kernel=cv2.getGaussianKernel(feature_width, 1))  # 这里可以高斯平滑一下,但是这个平滑反而让正确率变低了
    theda = np.arctan2(dy, dx) + np.pi  # 这里注意是dy,dx.注意这个区间是-pi,pi,先搞成正数
    theda = np.floor_divide(theda, np.pi / 4)
    theda = theda.astype(np.int)
    theda[theda >= how_many_orientation] = how_many_orientation - 1


    fv = []
    for fp in range(how_many_features):
        histogram = np.zeros((cell_length, cell_length, how_many_orientation))
        second_dim_start = x[fp] - feature_width // 2
        first_dim_start = y[fp] - feature_width // 2
        for cur_x in range(second_dim_start, second_dim_start + feature_width):
            for cur_y in range(first_dim_start, first_dim_start + feature_width):
                if 0 <= cur_x < image.shape[1] and 0 <= cur_y < image.shape[0]: # 这样写直接让我在很多时候没有看见报错
                    cur_theda = theda[cur_y, cur_x]  # 求出当前梯度指向哪个方向
                    histogram[(cur_y - first_dim_start) // each_cell_influence_length,
                              (cur_x - second_dim_start) // each_cell_influence_length, cur_theda] += \
                        magnitudes[cur_y, cur_x]
        fv.append(histogram.flatten())
    fv = np.array(fv)

    # 这之后一定要归一化https://blog.csdn.net/qq_40369926/article/details/88597406，归一化参考这个csdn
    divide_sum = np.sqrt(np.sum(np.square(fv), axis=1))
    # print(divide_sum)
    fv = fv / np.repeat(divide_sum.reshape(-1, 1), 128, axis=1)
    fv[fv > 0.2] = 0.2
    divide_sum = np.sqrt(np.sum(np.square(fv), axis=1))
    fv = fv / np.repeat(divide_sum.reshape(-1, 1), 128, axis=1)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv

