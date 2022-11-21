




import cv2
import numpy as np
import os.path as osp
import pickle
from random import shuffle
import matplotlib.pyplot as plt
from utils import *
import student_code_12012530 as sc

# 这个作业主要就是实现几个方法，然后在这个小数据集上面测试

# This is the list of categories / directories to use. The categories are
# somewhat sorted by similarity so that the confusion matrix looks more
# structured (indoor and then urban and then rural).
categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial', 'Suburb',
              'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry', 'Coast',
              'Mountain', 'Forest'];
# This list of shortened category names is used later for visualization
abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst',
                   'Mnt', 'For'];

# Number of training examples per category to use. Max is 100. For
# simplicity, we assume this is the number of test cases per category, as
# well.
num_train_per_cat = 100

# This function returns lists containing the file path for each train
# and test image, as well as lists with the label of each train and
# test image. By default all four of these lists will have 1500 elements
# where each element is a string.
data_path = osp.join('..', 'data')
print("over")
train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path,
                                                                                 categories,
                                                                                 num_train_per_cat);


def compute_acc(y_true, y_pred):
    from sklearn.metrics import confusion_matrix,accuracy_score
    acc = accuracy_score(y_true,y_pred)
    return acc


def K_cross_validation():
    from sklearn.model_selection import KFold
    the_best_k = -1
    the_max_acc = 0
    for k_current in range(1, 8):
        kf = KFold(n_splits=5)
        acc_list = []
        for train_index, test_index in kf.split(train_image_feats):
            # print(train_image_feats[train_index])
            predicted_categories = sc.nearest_neighbor_classify(train_image_feats[train_index],
                                                                train_labels_array[train_index],
                                                                train_image_feats[test_index],
                                                                 k=k_current)
            cur_acc = compute_acc(predicted_categories, train_labels_array[test_index])
            acc_list.append(cur_acc)
        mean_acc = np.mean(np.array(acc_list))
        if mean_acc > the_max_acc:
            the_max_acc = mean_acc
            the_best_k = k_current
    return the_best_k
if __name__ == "__main__":

    print(train_image_paths)
    print('Using SVM classifier to predict test set categories')

    train_image_feats = sc.get_tiny_images(train_image_paths)
    test_image_feats = sc.get_tiny_images(test_image_paths)
    predicted_categories = sc.svm_classify(train_image_feats, train_labels, test_image_feats)
    train_labels_array = np.array(train_labels)





    print(K_cross_validation())