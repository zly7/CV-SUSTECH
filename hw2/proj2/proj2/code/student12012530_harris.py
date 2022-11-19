from random import gauss
from ssl import RAND_add
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    ### 普通方法就是har是localmax,并且大于一个阈值10
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    gauss_kernel_size = 3
    gauss_kernel_sigma = 1
    m = image.shape[0]
    n = image.shape[1]
    aphla = 0.06
    # 1. Image derivatives
    Ix = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0)
    Iy = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1)
    # 2. Square of derivatives
    Ixx = np.multiply(Ix, Ix)
    Iyy = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)
    # 3.gaussion  他这个高斯平滑的原因在于有一些特征如果非常尖锐，则这个值会比较小
    gaussion_kernel = cv2.getGaussianKernel(ksize=gauss_kernel_size, sigma=gauss_kernel_sigma)
    Ixx = cv2.filter2D(Ixx, ddepth=-1, kernel=gaussion_kernel)  # 这个函数相当于apply任意的函数
    Ixy = cv2.filter2D(Ixy, ddepth=-1, kernel=gaussion_kernel)
    Iyy = cv2.filter2D(Iyy, ddepth=-1, kernel=gaussion_kernel)

    har = Ixx * Iyy - np.square(Ixy) - aphla * np.square(Ixx + Iyy)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    # 
    # 这个英文描述的非常好说白了要找到一个合适的半径让特征点正好是1500最开始的值半径应该是无穷大
    # #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################
    dim1_length = image.shape[0]
    dim2_length = image.shape[1]
    max_fetch_point = 1500
    early_stop = 7000


    def compute_radius_direct(x1,y1, flatten_index2, dim2_length):


        x2 = flatten_index2 // dim2_length
        y2 = flatten_index2 % dim2_length
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    max_radius = min(image.shape[0], image.shape[1])  # 第一个点的radius
    har_flatten = har.reshape(-1)
    sort_index = np.argsort(har_flatten)
    sort_index = np.flipud(sort_index)

    radii_array = [max_radius]
    dim1_array = [sort_index[0] // dim2_length]
    dim2_array = [sort_index[0] % dim2_length]
    har_mean = np.mean(har_flatten)
    for cur_index in sort_index[1:early_stop]:  # 这个curindex就是harflatten的值
        if har_flatten[cur_index] < 3 * har_mean: # 这种直接跳过
            continue
        cur_dim1 = cur_index // dim2_length
        cur_dim2 = cur_index % dim2_length
        current_value = 0.9 * har_flatten[cur_index]
        where_index = np.where(current_value < har_flatten)  # 这些值都大于1.1倍原本
        # npwhere这里返回tuple,index是harflatten的
        where_index = where_index[0]
        cur_radius_array = []
        for i_ in where_index:
            if i_ == cur_index:
                continue  # 本身这个点跳过
            cur_radius_array.append(compute_radius_direct(cur_dim1,cur_dim2, i_, dim2_length))

        cur_final_radius = np.min(np.array(cur_radius_array))  # 所有不符合要求的点最近的
        radii_array.append(cur_final_radius)
        dim1_array.append(cur_dim1)
        dim2_array.append(cur_dim2)
    radii_array = np.array(radii_array, dtype=np.float32)  # 半径肯定是小数
    radii_sort_index = np.argsort(radii_array)
    radii_sort_index = np.flipud(radii_sort_index)
    if len(radii_array) > max_fetch_point:
        final_radius = radii_array[radii_sort_index[max_fetch_point]]
    else:
        final_radius = 0  # 不对半径进行限制
    final_dim1 = []
    final_dim2 = []
    for cur_index in range(len(radii_array)):
        if radii_array[cur_index] > final_radius:
            final_dim1.append(dim1_array[cur_index])
            final_dim2.append(dim2_array[cur_index])
    final_dim1 = np.array(final_dim1)
    final_dim2 = np.array(final_dim2)
    return final_dim2, final_dim1, confidences, scales, orientations

