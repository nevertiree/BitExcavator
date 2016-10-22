import numpy as np
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score


# Data Preprocessing include loading data, fixing the missing value and reducing dimensionality

def get_csv_data(csv_path):
    # csv_object = csv.reader(open(csv_path, 'r'), delimiter=',')
    # csv_list = list(csv_object)
    # return np.array(csv_list[1:])
    raw_data = pd.read_csv(csv_path, skiprows=0, sep=",")
    print "Load the raw data, start transforming type..."
    return np.array(raw_data)


# Reducing dimensionality with Principal Component Analysis
def pca_by_number(origin_data, feature_number=9999999):
    # calculate the mean of the original matrix and remove it.
    mean = np.mean(origin_data, axis=0)
    print "Mean calculating is done ."
    mean_removed_matrix = origin_data - mean
    print "Mean removing is done ."

    # calculate the covariance matrix.
    covariance_matrix = np.cov(mean_removed_matrix, rowvar=0)
    print "Covariance calculating is done ."

    # calculate the eigenvalues of the covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eig(np.mat(covariance_matrix))
    print "Eigenvalues calculating is done ."

    # sort the eigenvalues from largest to smallest.
    eigenvalues_sorted_indice = np.argsort(eigenvalues)
    print "Eigenvalues sorting is done."

    # pick up the top N eigenvalue.
    eigenvalues_picked_indice = eigenvalues_sorted_indice[:-(feature_number + 1):-1]
    print "Picking up the top N eigenvalues."

    # transform the original data into the new space with N features.
    eigenvector_picked = eigenvectors[:, eigenvalues_picked_indice]
    print "Picking up the relative eigenvectors."

    # reconstruct the original data and return it for debug along with the reduced dimension data.
    low_dimension_matrix = mean_removed_matrix * eigenvector_picked

    # print out the importance of eigenvalue with matplotlib figure
    total_eigenvalue = np.sum(eigenvalues)
    eigen_importance = eigenvalues / total_eigenvalue
    print "Eigenvalue importance calculating is done."
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(eigenvalues_sorted_indice, eigen_importance[eigenvalues_sorted_indice])
    plt.show()

    # reconstruct_matrix = (low_dimension_matrix * eigenvector_picked.transpose()) + mean
    return low_dimension_matrix


def pca_by_lossy(origin_data, feature_lossy=0.1):
    # calculate the mean of the original matrix and remove it.
    mean = np.mean(origin_data, axis=0)
    print "Mean calculating is done ."
    mean_removed_matrix = origin_data - mean
    print "Mean removing is done ."

    # calculate the covariance matrix.
    covariance_matrix = np.cov(mean_removed_matrix, rowvar=0)
    print "Covariance calculating is done ."

    # calculate the eigenvalues of the covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eig(np.mat(covariance_matrix))
    print "Eigenvalues calculating is done ."

    # sort the eigenvalues from largest to smallest.
    eigenvalues_sorted_indice = np.argsort(eigenvalues)
    print "Eigenvalues sorting is done."

    # print out the importance of eigenvalue with matplotlib figure
    total_eigenvalue = np.sum(eigenvalues)
    eigen_importance = eigenvalues / total_eigenvalue
    print "Eigenvalue importance calculating is done."

    # find the threshold of lossy
    total_importance = 0
    threshold_number = 1
    print "Start calculate the threshold number"
    while (total_importance <= (1 - feature_lossy)) | (total_importance >= 1):
        total_importance += eigen_importance[threshold_number]
        threshold_number += 1
    print "Done calculate threshold number"

    # pick up the top N eigenvalue.
    eigenvalues_picked_indice = eigenvalues_sorted_indice[:-(threshold_number + 1):-1]
    print "Picking up the top N eigenvalues."

    # transform the original data into the new space with N features.
    eigenvector_picked = eigenvectors[:, eigenvalues_picked_indice]
    print "Picking up the relative eigenvectors."

    # reconstruct the original data and return it for debug along with the reduced dimension data.
    low_dimension_matrix = mean_removed_matrix * eigenvector_picked

    # show the plot
    # figure = plt.figure()
    # ax = figure.add_subplot(111)
    # ax.scatter(eigenvalues_sorted_indice, eigen_importance[eigenvalues_sorted_indice])
    # plt.show()

    # reconstruct_matrix = (low_dimension_matrix * eigenvector_picked.transpose()) + mean
    return low_dimension_matrix, total_importance, threshold_number


testing_data_path = "E:\\WorkSpace\\Kaggle\\2.DigitRecognizer\\DataSet\\test.csv"
origin = np.mat(get_csv_data(testing_data_path))

lossy = 0.08
dimension_reduction_matrix1, important1, feature_number1 = pca_by_lossy(origin, lossy)
print "Lossy = %f" % lossy
print important1
print feature_number1


def split_feature_target(training_data):
    feature = training_data[:, 1:]
    target = training_data[:, 0]
    return feature, target
