import math
import subprocess
from copy import deepcopy

import numpy as np
import sklearn

from KNN import KNNClassifier
from utils import *

target_attribute = 'Outcome'


def run_knn(k, x_train, y_train, x_test, y_test, formatted_print=True):
    neigh = KNNClassifier(k=k)
    neigh.train(x_train, y_train)
    y_pred = neigh.predict(x_test)
    acc = accuracy(y_test, y_pred)
    print(f'{acc * 100:.2f}%' if formatted_print else acc)


def get_top_b_features(x, y, b=5, k=51):
    """
    :param k: Number of nearest neighbors.
    :param x: array-like of shape (n_samples, n_features).
    :param y: array-like of shape (n_samples,).
    :param b: number of features to be selected.
    :return: indices of top 'b' features, sorted.
    """
    # TODO: Implement get_top_b_features function
    #   - Note: The brute force approach which examines all subsets of size `b` will not be accepted.

    assert 0 < b < x.shape[1], f'm should be 0 < b <= n_features = {x.shape[1]}; got b={b}.'
    top_b_features_indices = []

    # ====== YOUR CODE: ======
    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')
    max_acc = -np.inf
    test_counter = 0
    top_b_features_indices = []
    prev_iteration_accuracies = []
    while len(top_b_features_indices) < b:
        considered_feature_index = -1
        curr_feature_best_acc = -np.inf
        for i in range(8):
            if i in top_b_features_indices:
                continue
            test_counter += 1
            temp_top_b_features = deepcopy(top_b_features_indices)
            currently_tested_feature_index = i
            temp_top_b_features.append(currently_tested_feature_index)
            temp_top_b_features.sort()
            x_train_new = x_train[:, temp_top_b_features]
            y_train_new = y_train
            current_acc = get_k_fold_accuracy_for_feature_set(x_train_new, y_train_new, k, temp_top_b_features)
            prev_iteration_accuracies.append({i: current_acc})
            if current_acc > curr_feature_best_acc:
                curr_feature_best_acc = current_acc
                considered_feature_index = i
        if curr_feature_best_acc > max_acc:
            max_acc = curr_feature_best_acc
            top_b_features_indices.append(considered_feature_index)
        else:
            best_of_not_helping_acc = -np.inf
            best_of_not_helping_index = -1
            for acc in prev_iteration_accuracies:
                for k, v in acc.items():
                    if v > best_of_not_helping_acc and k not in top_b_features_indices:
                        best_of_not_helping_acc = v
                        best_of_not_helping_index = k
            top_b_features_indices.append(best_of_not_helping_index)

    # TODO: remove print
    print(top_b_features_indices)
    print(f'We tested {test_counter} sub-groups, total number of sub groups is: {math.comb(8, b)}')
    # ========================

    return top_b_features_indices


def find_best_b():
    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')
    best_k = 51
    for b in range(1, 8):
        print(f'best for b: {b}')
        top_m = get_top_b_features(x_train, y_train, b=b, k=best_k)
        x_train_new = x_train[:, top_m]
        x_test_test = x_test[:, top_m]
        exp_print(f'KNN in selected feature data: ')
        run_knn(best_k, x_train_new, y_train, x_test_test, y_test)


def get_k_fold_accuracy_for_feature_set(x_train, y_train, k=51, features_indices=None):
    kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=ID)
    accuracy_per_split = np.array([])
    # Perform K-fold CV over only the training dataset
    # Generate a split of data, train on the major split, then test on the smaller split
    for train_indexes, test_indexes in kf.split(x_train):
        neigh = KNNClassifier(k=k)
        neigh.train(x_train[train_indexes], y_train[train_indexes])
        y_pred = neigh.predict(x_train[test_indexes])
        acc = accuracy(y_train[test_indexes], y_pred)
        accuracy_per_split = np.append(accuracy_per_split, acc)
    avg_acc = np.mean(accuracy_per_split)
    # print(f'{avg_acc * 100:.2f}%, {features_indices}')

    return avg_acc


def run_cross_validation():
    """
    cross validation experiment, k_choices = [1, 5, 11, 21, 31, 51, 131, 201]
    """
    file_path = str(pathlib.Path(__file__).parent.absolute().joinpath("KNN_CV.pyc"))
    subprocess.run(['python', file_path])


def exp_print(to_print):
    print(to_print + ' ' * (30 - len(to_print)), end='')


# ========================================================================
if __name__ == '__main__':
    """
       Usages helper:
       (*) cross validation experiment
            To run the cross validation experiment over the K,Threshold hyper-parameters
            uncomment below code and run it
    """
    # run_cross_validation()

    # # ========================================================================

    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')

    best_k = 51
    b = 2

    # # ========================================================================

    print("-" * 10 + f'k  = {best_k}' + "-" * 10)
    exp_print('KNN in raw data: ')
    run_knn(best_k, x_train, y_train, x_test, y_test)

    top_m = get_top_b_features(x_train, y_train, b=b, k=best_k)
    x_train_new = x_train[:, top_m]
    x_test_test = x_test[:, top_m]
    exp_print(f'KNN in selected feature data: ')
    run_knn(best_k, x_train_new, y_train, x_test_test, y_test)
