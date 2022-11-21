import cv2
import numpy as np
import pickle

from tqdm import tqdm

from utils import load_image, load_image_gray
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
from PIL import Image
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans, kmeans_quantize
from time import time


def get_tiny_images(image_paths):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    To build a tiny image feature, simply resize the original image to a very
    small square resolution, e.g. 16x16. You can either resize the images to
    square while ignoring their aspect ratio or you can crop the center
    square portion out of each image. Making the tiny images zero mean and
    unit length (normalizing them) will increase performance modestly.

    Useful functions:
    -   cv2.resize
    -   use load_image(path) to load a RGB images and load_image_gray(path) to
        load grayscale images

    Args:
    -   image_paths: list of N elements containing image paths

    Returns:
    -   feats: N x d numpy array of resized and then vectorized tiny images
              e.g. if the images are resized to 16x16, d would be 256
    """
    # dummy feats variable

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    feats = []
    for image_p in image_paths:
        img = load_image_gray(image_p)
        img = cv2.resize(img, (16, 16))
        img = np.array(img).flatten()
        # print(img)
        # break
        img = img - img.mean()#这么标准化显然有问题
        img = img / np.linalg.norm(img)
        feats.append(img)
    feats = np.array(feats)
    print("make feats numpy array")
    print(type(feats))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def build_vocabulary(image_paths, vocab_size, dsift_size=15, dsift_step=25, number_of_patch = 5):
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
            -  frames is a N x 2 matrix of locations, which can be thrown away
            here (but possibly used for extra credit in get_bags_of_sifts if
            you're making a "spatial pyramid").
            -  descriptors is a N x 128 matrix of SIFT features
          Note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster to
          compute. Also, be sure not to use the default value of step size. It
          will be very slow and you'll see relatively little performance gain
          from extremely dense sampling. You are welcome to use your own SIFT
          feature code! It will probably be slower, though.
    -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
            http://www.vlfeat.org/matlab/vl_kmeans.html
              -  X is a N x d numpy array of sampled SIFT features, where N is
                 the number of features sampled. N should be pretty large!
              -  K is the number of clusters desired (vocab_size)
                 cluster_centers is a K x d matrix of cluster centers. This is
                 your vocabulary.

    Args:
    -   image_paths: list of image paths.
    -   vocab_size: size of vocabulary

    Returns:
    -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
        cluster center / visual word
    """
    # Load images from the training set. To save computation time, you don't
    # necessarily need to sample from all images, although it would be better
    # to do so. You can randomly sample the descriptors from each image to save
    # memory and speed up the clustering. Or you can simply call vl_dsift with
    # a large step size here, but a smaller step size in get_bags_of_sifts.
    #
    # For each loaded image, get some SIFT features. You don't have to get as
    # many SIFT features as you will in get_bags_of_sift, because you're only
    # trying to get a representative sample here.
    #
    # Once you have tens of thousands of SIFT features from many training
    # images, cluster them with kmeans. The resulting centroids are now your
    # visual word vocabulary.

    # length of the SIFT descriptors that you are going to compute.
    dim = 128
    d_a = []
    vocab = np.zeros((vocab_size, dim))
    index = 0
    for image_p in tqdm(image_paths):
        index = index + 1
        if index >= number_of_patch:  # 做一个跳过，不然太密集
            index = 0
        else:
            continue
        img = load_image_gray(image_p)
        frames, descriptors = dsift(img, fast=True, step=dsift_step, size=dsift_size)
        # print(descriptors.dtype) #uint8
        descriptors = descriptors.tolist()
        d_a.extend(descriptors)
    d_a = np.array(d_a)
    d_a = d_a.astype(np.float)
    print(f"d_a shape : {d_a.shape}")
    print(f"d_a dtype : {d_a.dtype}")

    vocab = kmeans(data=d_a, num_centers=vocab_size)  # 代码没有被限制在这里
    assert vocab.shape == (vocab_size, dim)
    return vocab

    # dim = 128
    # #vocab = np.zeros((vocab_size, dim))
    # des = np.zeros((1,128))
    # for i in tqdm(image_paths):
    #     image = load_image_gray(i)
    #     #descriptors N*128
    #     frames, descriptors = dsift(image,fast=True,step=25,size=50)
    #     des = np.append(des, descriptors , axis = 0)
    # des = des[1:]
    # vocab = kmeans(des, vocab_size)
    # return vocab


def get_bags_of_sifts(image_paths, vocab_filename, dsift_size=15, dsift_step=25):
    """
    This feature representation is described in the handout, lecture
    materials, and Szeliski chapter 14.
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
            http://www.vlfeat.org/matlab/vl_dsift.html
          frames is a M x 2 matrix of locations, which can be thrown away here
            (but possibly used for extra credit in get_bags_of_sifts if you're
            making a "spatial pyramid").
          descriptors is a M x 128 matrix of SIFT features
            note: there are step, bin size, and smoothing parameters you can
            manipulate for dsift(). We recommend debugging with the 'fast'
            parameter. This approximate version of SIFT is about 20 times faster
            to compute. Also, be sure not to use the default value of step size.
            It will be very slow and you'll see relatively little performance
            gain from extremely dense sampling. You are welcome to use your own
            SIFT feature code! It will probably be slower, though.
    -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
            finds the cluster assigments for features in data
              -  data is a M x d matrix of image features
              -  vocab is the vocab_size x d matrix of cluster centers
              (vocabulary)
              -  assignments is a Mx1 array of assignments of feature vectors to
              nearest cluster centers, each element is an integer in
              [0, vocab_size)

    Args:
    -   image_paths: paths to N images
    -   vocab_filename: Path to the precomputed vocabulary.
            This function assumes that vocab_filename exists and contains an
            vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
            or visual word. This ndarray is saved to disk rather than passed in
            as a parameter to avoid recomputing the vocabulary every run.

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
            feature representation. In this case, d will equal the number of
            clusters or equivalently the number of entries in each image's
            histogram (vocab_size) below.
    """
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = []
    vocab_size = len(vocab)

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    for image_p in tqdm(image_paths):
        # t1 = time()
        histogram = np.zeros(vocab_size)
        img = load_image_gray(image_p)
        frames, descriptors = dsift(img, fast=True, step=dsift_step, size=dsift_size)  # 这里肯定需要和你取这个值是一样的,被限制在这，结果忘记fast了
        # print(t1 - time());t1 = time()
        descriptors = descriptors.astype(np.float)
        assign_ = kmeans_quantize(descriptors, vocab)
        # print(t1 - time());t1 = time()
        for assign_value in assign_:
            histogram[assign_value] = histogram[assign_value] + 1
        # print(t1 - time());t1 = time()
        # histogram = histogram / np.max(histogram)
        # histogram, bin_edge = np.histogram(assign_, range(vocab_size + 1))
        histogram = histogram / np.linalg.norm(histogram)  # 标准化向量是指标准化之后向量模长是1
        feats.append(histogram)
    feats = np.array(feats)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
                              metric='euclidean', k=5):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which will increase
    performance (although you need to pick a reasonable value for k).

    Useful functions:
    -   D = sklearn_pairwise.pairwise_distances(X, Y)
          computes the distance matrix D between all pairs of rows in X and Y.
            -  X is a N x d numpy array of d-dimensional features arranged along
            N rows
            -  Y is a M x d numpy array of d-dimensional features arranged along
            N rows
            -  D is a N x M numpy array where d(i, j) is the distance between row
            i of X and row j of Y

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating
            the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    -   metric: (optional) metric to be used for nearest neighbor.
            Can be used to select different distance functions. The default
            metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
            well for histograms

    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """

    # - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
    #   'manhattan']. These metrics support sparse matrix
    #   inputs.
    def find_the_max_label(dic_t: dict):
        max_ = 0
        re_label = None
        for key in dic_t.keys():
            if dic_t[key] > max_:
                max_ = dic_t[key]
                re_label = key
        if re_label is None:
            raise "error in find_the_max_label"
        return re_label

    def assign_zero(dic_t: dict):
        for key in dic_t.keys():
            dic_t[key] = 0
        return

    test_labels = []
    label_dic = {}
    for ele in np.unique(train_labels):
        label_dic[ele] = 0
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    D = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats, metric=metric)
    for array_t in D:
        assert len(array_t) == len(train_image_feats)
        index_t = np.argsort(array_t)
        for i in index_t[:k]:
            label_dic[train_labels[i]] = label_dic[train_labels[i]] + 1
        test_labels.append(find_the_max_label(label_dic))
        assign_zero(label_dic)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels


def svm_classify(train_image_feats, train_labels, test_image_feats, penalty='l2',
                 loss='hinge', svm_regular_parameter=1.0):
    """
    This function will train a linear SVM for every category (i.e. one vs all)
    and then use the learned linear classifiers to predict the category of
    every test image. Every test feature will be evaluated with all 15 SVMs
    and the most confident SVM will "win". Confidence, or distance from the
    margin, is W*X + B where '*' is the inner product or dot product and W and
    B are the learned hyperplane parameters.  todo：左边这句话非常误导

    Useful functions:
    -   sklearn LinearSVC
          http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    -   svm.fit(X, y)
    -   set(l)

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating the
            ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    ## penalty : {'l1', 'l2'}, default='l2'
    #     Specifies the norm used in the penalization. The 'l2'
    #     penalty is the standard used in SVC. The 'l1' leads to ``coef_``
    #     vectors that are sparse.
    #
    # loss : {'hinge', 'squared_hinge'}, default='squared_hinge'
    #     Specifies the loss function. 'hinge' is the standard SVM loss
    #     (used e.g. by the SVC class) while 'squared_hinge' is the
    #     square of the hinge loss. The combination of ``penalty='l1'``
    #     and ``loss='hinge'`` is not supported.
    # C : float, default=1.0
    #     Regularization parameter. The strength of the regularization is
    #     inversely proportional to C. Must be strictly positive.
    # categories
    categories = list(set(train_labels))
    print(categories)
    # construct 1 vs all SVMs for each category
    svms = {cat: LinearSVC(random_state=0, tol=1e-3, loss=loss, penalty=penalty, C=svm_regular_parameter)
            for cat in categories}

    def find_the_max_label(dic_t: dict):
        max_ = -99999
        re_label = None
        for key in dic_t.keys():
            if dic_t[key] > max_:
                max_ = dic_t[key]
                re_label = key
        if re_label is None:
            raise "error in find_the_max_label"
        return re_label

    def assign_zero(dic_t: dict):
        for key in dic_t.keys():
            dic_t[key] = 0
        return

    test_labels = []
    label_dic = {}
    for cat in categories:
        label_dic[cat] = -1
    for cat in categories:
        print(f"current cat : {cat}")
        # current_label = np.copy(train_labels)
        current_label = np.zeros(len(train_labels), dtype=np.int)
        # current_label[train_labels == cat] = 1 # 这代码很疑惑
        for i in range(len(current_label)):
            if train_labels[i] == cat:
                current_label[i] = 1
        # print(np.sum(current_label))
        # current_label[current_label != 1] = 0
        svms[cat].fit(train_image_feats, current_label)
    for t in test_image_feats:
        for key in svms:
            label_dic[key] = svms[key].decision_function(t.reshape(1, -1))
        test_labels.append(find_the_max_label(label_dic))
        assign_zero(label_dic)

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels
