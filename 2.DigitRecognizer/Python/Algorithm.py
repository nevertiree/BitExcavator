import numpy as np
import time
import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

model = RandomForestClassifier(n_estimators=100)


def practice_algorithm(model, feature, target):

    # split arrays or matrices into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2,
                                                                              random_state=42)
    # Training the model and counting the time
    start_time = int(time.time())
    model.fit(X_train,y_train)
    print("Algorithm Consume Time: %f" % (time.time() - start_time))

    # Count the percsion of the model
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print ("Model Precision : %0.3f (+/- %0.3f) ") % (scores.mean(), scores.std())

    # predicted and expected value
    predicted = model.predict(y_train)
    print metrics.classification_report(y_test, predicted)

    # recall
    m_recall = metrics.recall_score(y_test, predicted, average='macro')
    print ("Model Recall : %0.3f") % m_recall

    # F1 value
    f_measure = scores.mean() * m_recall * 2 / (scores.mean() + m_recall)
    # f_measure = metrics.f1_score(test_target, average='macro')
    print ("F1 Value : %s ") % f_measure

    # importance
    # print model.feature_importances_


def print_predict_result(model, X_train, y_train, X_test, sample_output):

    # Training the model
    start_time = int(time.time())
    model.fit(X_train, y_train)
    print("Algorithm Consume Time: %f" % (time.time() - start_time))

    predicted = model.predict(X_test)
    image_id = sample_output[:,0]

    result = zip(image_id, predicted)

    print type(predicted)
    print type(image_id)
    print type(result)
    print result

    #predicted.tofile("result.csv",sep=",",format="%s")
    # with open('result.csv', 'wb') as res:
    #     res.write((",".join(result)).encode("utf-8") + "\n")
    #     # np.savetxt(result, sample_output, delimiter=",", newline='\n', fmt='%d')
    #np.ndarray.tofile(sample_output, "result.csv", sep=',')
    writer = csv.writer(file('result.csv', 'wb'))
    for line in result:
        writer.writerow(line)
