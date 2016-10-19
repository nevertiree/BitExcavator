import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics

#Cluster Algorithm Version 1 :
#Using 02 Survived      03 Pclass     13 Gender    06 Age     10 Fare   14 Embark

def run( feature_train, feature_test, target_train, target_test):

    # the number of the trees in the forest
    model = RandomForestClassifier(n_estimators=100)

    # Training the model and counting the time
    start_time = int(time.time())
    model.fit(feature_train, target_train)
    print("Algorithm Consume Time: %f" % (time.time() - start_time))

    # Count the percsion of the model
    scores = cross_val_score(model, feature_train, target_train, cv=5, scoring='accuracy')
    print ("Model Precision : %0.3f (+/- %0.3f) ") % (scores.mean(), scores.std())

    #predict
    pred = model.predict(feature_test)
    print metrics.classification_report(target_test, pred)

    #recall
    m_recall = metrics.recall_score(target_test, pred)
    print ("Model Recall : %0.3f") % m_recall

    #F1 value
    f_measure = metrics.f1_score(target_test, pred)
    print ("F1 Value : %s ")% f_measure

    #importance
    print model.feature_importances_
