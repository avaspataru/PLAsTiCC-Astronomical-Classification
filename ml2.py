import pandas as pd
import numpy as np
import gc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn import tree
from sklearn import preprocessing
from random import *
from helper import standardizeData,normalizeData,equalProbabilities
from sklearn.preprocessing import StandardScaler


def scale(dataFrame):
    dataFrame_new = dataFrame.copy()
    ss = StandardScaler()
    gc.collect()
    return ss.fit_transform(dataFrame_new)

def main():

    print "Reading data"
    training_set_raw = pd.read_csv('/modules/cs342/Assignment2/training_set.csv',header=0)
    training_set_metadata_raw = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv',header=0)

    #classes, not class 99
    training_set_targets = training_set_metadata_raw['target']
    classes = sorted(training_set_targets.unique())
    class_weight = {
    c: 1 for c in classes
    }
    for c in [64, 15]:
        class_weight[c] = 2

    print "Formatting data"
    training_set_data = training_set_metadata_raw.drop('target',axis=1)
    training_set_data = training_set_data.drop('distmod',axis=1)

    training_set_raw['flux_ratio_sq'] = np.power(training_set_raw['flux'] / training_set_raw['flux_err'], 2.0)
    training_set_raw['flux_by_flux_ratio_sq'] = training_set_raw['flux'] * training_set_raw['flux_ratio_sq']

    aggs = {
    'mjd': ['min', 'max', 'size'],
    'passband': ['min', 'max', 'mean', 'median', 'std'],
    'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
    'detected': ['mean'],
    'flux_ratio_sq':['sum','skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
    }

    agg_train = training_set_raw.groupby('object_id').agg(aggs)

    new_columns = [
    k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_train.columns = new_columns
    agg_train['mjd_diff'] = agg_train['mjd_max'] - agg_train['mjd_min']
    agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
    agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
    agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
    agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']

    del agg_train['mjd_max'], agg_train['mjd_min']
    agg_train.head()

    del training_set_raw
    gc.collect()

    full_train = agg_train.reset_index().merge(
    right=training_set_data, # this is without some cols
    how='outer',
    on='object_id'
    )

    full_train_ids = full_train['object_id']
    full_train = full_train.drop('object_id', axis=1)

    print full_train.columns

    print "Split extragalactic "
    extra = np.where(full_train['hostgal_photoz']==0.0)
    extragalactic_data = full_train.drop(full_train.index[extra])
    extragalactic_targets = training_set_targets.drop(training_set_targets.index[extra])

    extragalactic_data = scale(extragalactic_data)
    print "Model for extra:"
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    print cross_val_score(clf, extragalactic_data, extragalactic_targets, cv=10, scoring="neg_log_loss").mean()

    print "Split intragalactic "
    intra = np.where(full_train['hostgal_photoz']!=0.0)
    intragalactic_data = full_train.drop(full_train.index[intra])
    intragalactic_targets = training_set_targets.drop(training_set_targets.index[intra])

    intragalactic_data = scale(intragalactic_data)
    print "Model for intra:"
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    print cross_val_score(clf, intragalactic_data, intragalactic_targets, cv=10, scoring="neg_log_loss").mean()


    full_train = scale(full_train)
    print "Training model"
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    #clf.fit(training_set_data, training_set_targets.values.ravel())
    print cross_val_score(clf, full_train, training_set_targets, cv=10, scoring="neg_log_loss").mean()

    #test_set = pd.read_csv('/modules/cs342/Assignment2/test_set.csv',header=0)
    #test_set_metadata = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv',header=0)

    #ans = bcf.predict_proba(test_set.values)[:, 1]
    #df = pd.DataFrame(ans)
    #df.to_csv("test_set_predictions.csv", index = False, header = False)

main()
