import numpy as np

#the script writen by myself

import numpy as np

from ReadData import *
from CleanData import *
from ClusterAlgorithm import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics

# Open up the csv file in to a Python object
train_data_path = "E:\\WorkSpace\\MyKaggle\\Titanic\\DataSet\\trainingData.csv"
training_matrix = get_data_matrix(train_data_path)

test_data_path = "E:\\WorkSpace\\MyKaggle\\Titanic\\DataSet\\testingData.csv"
testing_matrix = get_data_matrix(test_data_path)

#clean the data which is default
calculate_average_age(training_matrix, 6)
calculate_average_age(testing_matrix, 5)

#get the targe and feature using the algorithm version 1

feature_train,target_train = target_feature_of_train_1(training_matrix)
feature_test = feature_of_test_1(testing_matrix)

#fix the model (cross valid)

#feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.3, random_state=43)

print ("RF:")

model = RandomForestClassifier(n_estimators=100)

# Training the model and counting the time
start_time = int(time.time())
model.fit(feature_train, target_train)
pred = model.predict(feature_test)

number = testing_matrix[:,1]

result = zip(number,pred)

# print np.array(result).transpose()



print result
# f = open("result.csv","w")
# f.write("\n".join(result))
# f.close()


writer = csv.writer(file('result.csv', 'wb'))
for line in result:
    writer.writerow(line)

# run( feature_train, feature_test, target_train, target_test)