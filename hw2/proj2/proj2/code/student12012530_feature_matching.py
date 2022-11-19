import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################

    threshold= 0.9
    matches = []
    confidences = []
    n = len(features1)
    m = len(features2)
    # print(features2.shape)
    for cur_index in range(n):
        cur_attr = features1[cur_index]
        distance_array = np.square(features2 - cur_attr)
        assert len(distance_array) == m
        distance_array = np.sqrt(np.sum(distance_array, axis= 1)) # 这边再求个和,这里不sqrt
        # print(distance_array.shape)
        sort_index = np.argsort(distance_array)
        if distance_array[sort_index[0]] < threshold * distance_array[sort_index[1]]: # distance是越小越好
                matches.append(np.array([cur_index,sort_index[0]]))
                confidences.append(distance_array[sort_index[1]]/distance_array[sort_index[0]])
    matches = np.array(matches)

    confidences = np.array(confidences)
    idxs = np.flipud(confidences.argsort())  # 这才是argsort的正确用法
    matches = matches[idxs]
    confidences = confidences[idxs]
    # 这里要按照confidence排序啊 这里很核心啊，这个函数应该没问题了

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences

