import pandas as pd
import numpy as np
import operator
import time
import gc
import math
from collections import deque
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
    df = dataFrame.copy()
    col = df.columns.values

    for c in col:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return dataFrame

#def scale(dataFrame):
#    dataFrame_new = dataFrame.copy()
#    ss = StandardScaler()
#    gc.collect()
#    return ss.fit_transform(dataFrame_new)


def splitGalaxies(dataFrame, targets):
    print "Split extragalactic "
    extra = np.where(dataFrame['hostgal_photoz']==0.0)
    extragalactic_data = dataFrame.drop(dataFrame.index[extra])
    extra_ids = extragalactic_data['object_id']
    extragalactic_data = extragalactic_data.drop('object_id',axis=1)
    extragalactic_targets = targets.drop(targets.index[extra])
    extragalactic_data = scale(extragalactic_data)


    print "Split intragalactic "
    intra = np.where(dataFrame['hostgal_photoz']!=0.0)
    intragalactic_data = dataFrame.drop(dataFrame.index[intra])
    intra_ids = intragalactic_data['object_id']
    intragalactic_data = intragalactic_data.drop('object_id',axis=1)
    intragalactic_targets = targets.drop(targets.index[intra])

    intragalactic_data = scale(intragalactic_data)
    #return extragalactic_data, extragalactic_targets, extra_ids, intragalactic_data, intragalactic_targets, intra_ids
    return extragalactic_data, extragalactic_targets,extra_ids, intragalactic_data, intragalactic_targets, intra_ids


def format(set_metadata_raw, set_raw):
    print "Formatting data"

    set_data = set_metadata_raw.drop('distmod',axis=1)

    set_raw['flux_ratio_sq'] = np.power(set_raw['flux'] / set_raw['flux_err'], 2.0)
    set_raw['flux_by_flux_ratio_sq'] = set_raw['flux'] * set_raw['flux_ratio_sq']

    aggs = {
    'mjd': ['min', 'max', 'size'],
    'passband': ['min', 'max', 'mean', 'median', 'std'],
    'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
    'detected': ['mean'],
    'flux_ratio_sq':['sum','skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
    }

    agg_train = set_raw.groupby('object_id').agg(aggs)

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

    del set_raw
    gc.collect()

    full_train = agg_train.reset_index().merge(
    right=set_data, # this is without some cols
    how='outer',
    on='object_id'
    )

    return full_train

def get_objects_by_id(path, chunksize=1000000):
    """
    Generator that iterates over chunks of PLAsTiCC Astronomical Classification challenge
    data contained in the CVS file at path.

    Yields subsequent (object_id, pd.DataFrame) tuples, where each DataFrame contains
    all observations for the associated object_id.

    Inputs:
        path: CSV file path name
        chunksize: iteration chunk size in rows

    Output:
        Generator that yields (object_id, pd.DataFrame) tuples
    """
    # set initial state
    last_id = None
    last_df = pd.DataFrame()

    for df in pd.read_csv(path, chunksize=chunksize):

        # Group by object_id; store grouped dataframes into dict for fast access
        grouper = {
            object_id: pd.DataFrame(group)
            for object_id, group in df.groupby('object_id')
        }

        # queue unique object_ids, in order, for processing
        object_ids = df['object_id'].unique()
        queue = deque(object_ids)

        # if the object carried over from previous chunk matches
        # the first object in this chunk, stitch them together
        first_id = queue[0]
        if first_id == last_id:
            first_df = grouper[first_id]
            last_df = pd.concat([last_df, first_df])
            grouper[first_id] = last_df
        elif last_id is not None:
            # save last_df and return as first result
            grouper[last_id] = last_df
            queue.appendleft(last_id)

        # save last object in chunk
        last_id = queue[-1]
        last_df = grouper[last_id]

        # check for edge case with only one object_id in this chunk
        if first_id == last_id:
            # yield nothing for now...
            continue

        # yield all but last object, which may be incomplete in this chunk
        while len(queue) > 1:
            object_id = queue.popleft()
            object_df = grouper.pop(object_id)
            yield (object_id, object_df)

    # yield remaining object
    yield (last_id, last_df)


def fill_in_hostgal_specz(dataFrame):
    df = dataFrame.copy()

    df['hostgal_specz'] = df['hostgal_photoz']
    return df

def do_my_predict(extra_model, intra_model):
    print "some other stuff"
    print "Reading test data"
    test_set_metadata_raw = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')
    path = '/modules/cs342/Assignment2/test_set.csv'

     # set initial state
    last_id = None
    last_df = pd.DataFrame()

    for df in pd.read_csv(path, chunksize=chunksize):
        # Group by object_id; store grouped dataframes into dict for fast access
        grouper = {
            object_id: pd.DataFrame(group)
            for object_id, group in df.groupby('object_id')
        }

        # queue unique object_ids, in order, for processing
        object_ids = df['object_id'].unique()
        queue = deque(object_ids)

        # if the object carried over from previous chunk matches
        # the first object in this chunk, stitch them together
        first_id = queue[0]
        if first_id == last_id:
            first_df = grouper[first_id]
            last_df = pd.concat([last_df, first_df])
            grouper[first_id] = last_df
        elif last_id is not None:
            # save last_df and return as first result
            grouper[last_id] = last_df
            queue.appendleft(last_id)

        # save last object in chunk
        last_id = queue[-1]
        last_df = grouper[last_id]

        # check for edge case with only one object_id in this chunk
        if first_id == last_id:
            # yield nothing for now...
            continue
         # yield all but last object, which may be incomplete in this chunk
        while len(queue) > 1:
            object_id = queue.popleft()
            object_df = grouper.pop(object_id)
            yield (object_id, object_df)

     # yield remaining object
    yield (last_id, last_df)


    #print "Formatting test data"
    #test_set_targets = test_set_metadata_raw['target']

    #full_test = format(test_set_metadata_raw, test_set_raw)
    #extragalactic_test_data, extragalactic_test_targets, extra_test_ids,  intragalactic_test_data, intragalactic_test_targets, intra_test_ids = splitGalaxies(full_test, test_set_targets)

    #ans = bcf.predict_proba(test_set.values)[:, 1]
    #df = pd.DataFrame(ans)
    #df.to_csv("test_set_predictions.csv", index = False, header = False)

def main():

    mode = 1 #0-cv, 1-predict

    print "Reading train data"
    training_set_raw = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')
    training_set_metadata_raw = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')

    #classes, not class 99
    training_set_targets = training_set_metadata_raw['target']
    training_set_data = training_set_metadata_raw.drop('target',axis=1)

    classes = sorted(training_set_targets.unique())
    class_weight = {
    c: 1 for c in classes
    }
    for c in [64, 15]:
        class_weight[c] = 2


    full_train = format(training_set_data, training_set_raw)
    extragalactic_data, extragalactic_targets, extra_ids,  intragalactic_data, intragalactic_targets, intra_ids = splitGalaxies(full_train, training_set_targets)

    if mode==0:
        print "Model for extra:"
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        extragalactic_data = fill_in_hostgal_specz(extragalactic_data)
        print cross_val_score(clf, extragalactic_data, extragalactic_targets, cv=10, scoring="neg_log_loss").mean()

        print "Model for intra:"
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        intragalactic_data = fill_in_hostgal_specz(intragalactic_data)
        print cross_val_score(clf, intragalactic_data, intragalactic_targets, cv=10, scoring="neg_log_loss").mean()

        full_train_ids = full_train['object_id']
        full_train = full_train.drop('object_id', axis=1)
        full_train = scale(full_train)
        print "Training model"
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        #clf.fit(training_set_data, training_set_targets.values.ravel())
        print cross_val_score(clf, full_train, training_set_targets, cv=10, scoring="neg_log_loss").mean()

    else:
        print "Training"
        extra_model = RandomForestClassifier(n_jobs=2)
        extra_model.fit(extragalactic_data, extragalactic_targets.values.ravel())
        intra_model = RandomForestClassifier(n_jobs=2)
        intragalactic_data = fill_in_hostgal_specz(intragalactic_data)
        intra_model.fit(intragalactic_data, intragalactic_targets.values.ravel())
        print "Finished training. Starting predictions"

        print "Reading test data"
        test_set_metadata_raw = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')
        filepath = '/modules/cs342/Assignment2/test_set.csv'

        for object_id, df in get_objects_by_id(filepath):
            combined = format(fill_in_hostgal_specz(test_set_metadata_raw.loc[test_set_metadata_raw['object_id']==object_id]),df)
            combined = combined.drop('object_id',axis=1)
            combined = scale(combined)

            if object_id in extra_ids:
                print str(object_id) + " is extra"
                ans = extra_model.predict_proba(combined)[:,1]
            else:
                print str(object_id) + " is intra"
                ans = intra_model.predict_proba(combined)[:,1]
            print ans
        print "DONE."

main()
