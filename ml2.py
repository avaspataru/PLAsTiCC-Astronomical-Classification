import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import label_binarize
from sklearn import tree
from sklearn import preprocessing
from random import *
from helper import standardizeData,normalizeData,equalProbabilities

def main():

    print "Reading data"
    training_set_raw = pd.read_csv('/modules/cs342/Assignment2/training_set.csv',header=0)
    training_set_metadata_raw = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv',header=0)

    print "Formatting data"
    training_set_targets = training_set_metadata_raw['target']
    #training_set_targets = label_binarize(training_set_targets, classes=['class_6','class_15','class_16','class_42','class_52','class_53',
    #'class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])

    training_set_data = training_set_metadata_raw.drop('target',axis=1)
    training_set_data = training_set_data.drop('hostgal_specz',axis=1)
    training_set_data = training_set_data.drop('hostgal_photoz',axis=1)
    training_set_data = training_set_data.drop('distmod',axis=1)
    training_set_data = training_set_data.drop('mwebv',axis=1)

    print "Training model"
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=None, max_depth=4)
    #clf.fit(training_set_data, training_set_targets.values.ravel())
    print cross_val_score(clf, training_set_data, training_set_targets, cv=10, scoring="neg_log_loss").mean()

    #test_set = pd.read_csv('/modules/cs342/Assignment2/test_set.csv',header=0)
    #test_set_metadata = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv',header=0)

    #ans = bcf.predict_proba(test_set.values)[:, 1]
    #df = pd.DataFrame(ans)
    #df.to_csv("test_set_predictions.csv", index = False, header = False)

main()
