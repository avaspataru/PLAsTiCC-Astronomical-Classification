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

        extra_classes = extra_model.classes_
        intra_classes = intra_model.classes_
        column_names = []
        column_names.append('object_id')
        for classi in extra_classes:
            className = "class_" + str(classi)
            column_names.append(className)
        for classi in intra_classes:
            className = "class_" + str(classi)
            column_names.append(className)
        column_names.append("class_99")
        lst = []
        count = 0
        batch_no = 0
        batch_list = []
        print " >Starting new batch 0"
        for obj_id, d in get_objects_by_id(filepath):
            batch_list.append((obj_id,d))
            if(count == 2):
                print " >>Formatting batch objects"
                batch_extra_dataFrame = pd.DataFrame()
                batch_intra_dataFrame = pd.DataFrame()
                myextrabatchlist = []
                myintrabatchlist = []
                for object_id, df in batch_list:
                    combined = format(fill_in_hostgal_specz(test_set_metadata_raw.loc[test_set_metadata_raw['object_id']==object_id]),df)
                    #combined = combined.drop('object_id',axis=1)
                    combined = scale(combined) #combined is a df with one row for this object id
                    if (object_id in extra_ids):
                        myextrabatchlist.append(combined.values[0])
                    else:
                        myintrabatchlist.append(combined.values[0])
                batch_extra_dataFrame= pd.DataFrame(myextrabatchlist,columns=combined.columns)
                batch_intra_dataFrame= pd.DataFrame(myintrabatchlist,columns=combined.columns)
                extra_ans = []
                intra_ans = []
                objids = [[]]
                print " >>Predicting extra"
                if(len(batch_extra_dataFrame.index)>0):
                    objids1 = batch_extra_dataFrame['object_id'].values.tolist()

                    objids = []
                    for id in objids1:
                        l1 = [id]
                        print l1
                        objids.append(l1)
                        print objids
                    batch_extra_dataFrame = batch_extra_dataFrame.drop('object_id', axis=1)
                    extra_ans = extra_model.predict_proba(batch_extra_dataFrame)
                    z = np.zeros((len(extra_ans),6)) # zeros for intra classes and class 99
                    print extra_ans
                    print z
                    extra_ans = np.append(extra_ans,z,axis=1)
                    print extra_ans
                    print objids
                    extra_ans = np.append(objids,extra_ans,axis=1)
                print " >>Predicting intra"
                objids = [[]]
                if(len(batch_intra_dataFrame.index)>0):
                    objids1 = batch_intra_dataFrame['object_id'].values.tolist()

                    objids = []
                    for id in objids1:
                        l1 = [id]
                        print l1
                        objids.append(l1)
                        print objids
                    batch_intra_dataFrame = batch_intra_dataFrame.drop('object_id', axis=1)
                    intra_ans = intra_model.predict_proba(batch_intra_dataFrame)
                    z = np.zeros((len(intra_ans),10)) # zeros for extra classes and class 99
                    intra_ans = np.append(z,intra_ans,axis=1)
                    intra_ans = np.append(objids,intra_ans,axis=1)

                print extra_ans
                print intra_ans
                print " >>Putting together"
                arr = []
                if((len(batch_extra_dataFrame.index)>0) and (len(batch_intra_dataFrame.index)>0) ):
                    arr = np.concatenate((extra_ans,intra_ans), axis=0)
                else:
                    if (len(batch_intra_dataFrame.index)>0):
                        arr = intra_ans
                    else:
                        if (len(batch_extra_dataFrame.index)>0):
                            arr = extra_ans

                print " >>Write to csv"
                finish = pd.DataFrame(arr, columns=column_names)
                print finish
                print finish
                if(batch_no==0):
                    finish.to_csv("predictions.csv", index = False, header = True)
                else:
                    with open('predictions.csv', 'a') as f:
                        finish.to_csv(f, index = False, header=False)
                break

                #index = 0
                #row= []
                #row.append(object_id)
                #if object_id in extra_ids:
                #    ans = extra_model.predict_proba(combined)
                #    for classi in extra_classes:
                #        row.append(ans[:,index][0])
                #        index = index + 1
                #    for classi in intra_classes:
                #        row.append(0.0)

                #else:
                #    ans = intra_model.predict_proba(combined)
                #    for classi in extra_classes:
                #        row.append(0.0)
                #    for classi in intra_classes:
                #        row.append(ans[:,index][0])
                #        index = index + 1
                #row.append(0.0) #class99
                #lst.append(row)

                print " >Starting new batch" + str(batch_no + 1)
                batch_no = batch_no + 1
                lst = 0
                count = 0
                finish = pd.DataFrame(columns=column_names)

            else:
                count = count + 1
                if(count == 2500):
                    print "  >>25%..."
                if(count == 5000):
                    print "  >>50%..."
                if(count == 7500):
                    print "  >>75%..."
            if(batch_no == 3):
                break
        print "DONE."

main()
