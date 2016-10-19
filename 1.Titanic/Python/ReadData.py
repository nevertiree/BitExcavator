import csv as csv
import numpy as np

#training data
#train_data_csv_path = "/home/Lance/BitExcavator/Titanic/DataSet/trainingData.csv"
train_data_path = "E:\\WorkSpace\\MyKaggle\\Titanic\\DataSet\\trainingData.csv"

#testing data
#train_data_csv_path = "/home/Lance/BitExcavator/Titanic/DataSet/testingData.csv"
test_data_path = "E:\\WorkSpace\\MyKaggle\\Titanic\\DataSet\\testingData.csv"

# read the file from the disc
def get_data_matrix(path):
    csv_object = csv.reader(open(path,'rb'),delimiter=',')
    csv_list = list(csv_object)
    return np.array(csv_list)[1:]

#attach the training data header
# 00 ''         01 'PassengerId' 02 'Survived'  03 'Pclass'  04 'Name' 05 'Sex'
# 06 'Age'      07 'SibSp'       08 'Parch'     09 'Ticket'  10 'Fare' 11 'Cabin'
# 12 'Embarked' 13 'Gender'      14 'Embark'

#attach the testing data header
# 00 ''         01 'PassengerId' 02 'Pclass'  03 'Name'  04 'Sex'    05 'Age'
# 06 'SibSp'    07 'Parch'       08 'Ticket'  09 'Fare'  10 'Cabin'  11 'Embarked'
# 12 'Gender'   13 'Embark'

# input the raw matirx and output the useful columns
def target_feature_of_train_1(training_data):
    #get the special matrix from this algorithm version
    matirx = (training_data[:,2],training_data[:,3],training_data[:,13],training_data[:,6],training_data[:,10],training_data[:,14]) #tuple
    matirx = [map(float,row) for row in matirx] # transform the data into the float type
    matirx = np.array(matirx).transpose() # array -> transpose

    # get the target and the feature
    target = matirx[:, 0]
    feature = matirx[:, 1:]
    return feature,target

#Using 02 Pclass 12 Gender 05 Age 09 Fare  13 Embark
def feature_of_test_1(testing_data):
    matrix = (testing_data[:, 2], testing_data[:, 12], testing_data[:, 5], testing_data[:, 9], testing_data[:, 13])  # tuple
    matrix = [map(float, row) for row in matrix]  # transform the data into the float type
    matrix = np.array(matrix).transpose()  # array -> transpose
    # get the feature
    return matrix
