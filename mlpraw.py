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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn import tree
from sklearn import preprocessing
from random import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def scale(dataFrame):
    df = dataFrame.copy()
    col = df.columns.values

    for c in col:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return dataFrame


def splitGalaxies(dataFrame, targets):
    print "Split extragalactic "
    extra = np.where(dataFrame['hostgal_photoz']==0.0)
    extragalactic_data = dataFrame.drop(dataFrame.index[extra])
    extra_ids = extragalactic_data['object_id'].values.tolist()
    extragalactic_data = extragalactic_data.drop('object_id',axis=1)
    extragalactic_targets = targets.drop(targets.index[extra])


    print "Split intragalactic "
    intra = np.where(dataFrame['hostgal_photoz']!=0.0)
    intragalactic_data = dataFrame.drop(dataFrame.index[intra])
    intra_ids = intragalactic_data['object_id'].values.tolist()
    intragalactic_data = intragalactic_data.drop('object_id',axis=1)
    intragalactic_targets = targets.drop(targets.index[intra])

    return extragalactic_data, extragalactic_targets,extra_ids, intragalactic_data, intragalactic_targets, intra_ids


def splitTestGalaxies(dataFrame):
    print "Split extragalactic "
    extra = np.where(dataFrame['hostgal_photoz']==0.0)
    extragalactic_data = dataFrame.drop(dataFrame.index[extra])
    extra_ids = extragalactic_data['object_id'].values.tolist()

    print "Split intragalactic "
    intra = np.where(dataFrame['hostgal_photoz']!=0.0)
    intragalactic_data = dataFrame.drop(dataFrame.index[intra])
    intra_ids = intragalactic_data['object_id'].values.tolist()

    return extra_ids,intra_ids

def format(set_metadata_raw, set_raw):

    set_metadata_raw = fill_in_hostgal_specz(set_metadata_raw)
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

def my_predict(column_names,my_extra_data_list, my_intra_data_list, test_set_metadata_raw, extra_model, intra_model):

    formatted_columns = [u'object_id', u'mjd_size', u'flux_by_flux_ratio_sq_sum',
   u'flux_by_flux_ratio_sq_skew', u'flux_ratio_sq_sum',
   u'flux_ratio_sq_skew', u'flux_err_min', u'flux_err_max',
   u'flux_err_mean', u'flux_err_median', u'flux_err_std', u'flux_err_skew',
   u'flux_min', u'flux_max', u'flux_mean', u'flux_median', u'flux_std',
   u'flux_skew', u'detected_mean', u'passband_min', u'passband_max',
   u'passband_mean', u'passband_median', u'passband_std', u'mjd_diff',
   u'flux_diff', u'flux_dif2', u'flux_w_mean', u'flux_dif3', u'ra',
   u'decl', u'gal_l', u'gal_b', u'ddf', u'hostgal_specz',
   u'hostgal_photoz', u'hostgal_photoz_err', u'mwebv']

    finish = pd.DataFrame(columns=column_names)

    batch_extra_dataFrame = pd.DataFrame(columns = formatted_columns)
    batch_intra_dataFrame = pd.DataFrame(columns = formatted_columns)
    my_extra_data_batch = pd.DataFrame(columns = ['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected'])
    my_intra_data_batch = pd.DataFrame(columns = ['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected'])


    if(len(my_extra_data_list)>0):
        my_extra_data_batch = pd.concat(my_extra_data_list)
    if(len(my_intra_data_list)>0):
        my_intra_data_batch = pd.concat(my_intra_data_list)

    tt1 = test_set_metadata_raw.loc[test_set_metadata_raw['object_id'].isin(my_extra_data_batch['object_id'].values.tolist())]
    if(len(my_extra_data_batch.index)>0):
        batch_extra_dataFrame= format(tt1, my_extra_data_batch)
        scaler = StandardScaler()
        batch_extra_dataFrame.loc[:, batch_extra_dataFrame.columns != 'object_id' ] = (scaler.fit_transform(batch_extra_dataFrame.loc[:,batch_extra_dataFrame.columns != 'object_id' ]))
        #print batch_extra_dataFrame['object_id']
    else:
        batch_extra_dataFrame = pd.DataFrame(columns = formatted_columns)
    tt2 = test_set_metadata_raw.loc[test_set_metadata_raw['object_id'].isin(my_intra_data_batch['object_id'].values.tolist())]
    if(len(my_intra_data_batch.index)>0):
        batch_intra_dataFrame= format(tt2, my_intra_data_batch)
        scaler = StandardScaler()
        batch_intra_dataFrame.loc[:, batch_intra_dataFrame.columns != 'object_id' ] = (scaler.fit_transform(batch_intra_dataFrame.loc[:,batch_intra_dataFrame.columns != 'object_id' ]))
        #print batch_intra_dataFrame['object_id']
    else:
        batch_intra_dataFrame = pd.DataFrame(columns = formatted_columns)

    extra_ans = []
    intra_ans = []
    objids = [[]]
    print " >>Predicting extra"
    if(len(batch_extra_dataFrame.index)>0):
        objids1 = batch_extra_dataFrame['object_id'].values.tolist()
        objids = []
        for id in objids1:
            l1 = [id]
            objids.append(l1)
        batch_extra_dataFrame = batch_extra_dataFrame.drop('object_id', axis=1)
        extra_ans = extra_model.predict_proba(batch_extra_dataFrame)
        #print extra_model.classes_
        z = np.zeros((len(extra_ans),6)) # zeros for intra classes and class 99
        extra_ans = np.append(extra_ans,z,axis=1)
        extra_ans = np.append(objids,extra_ans,axis=1)
    print " >>Predicting intra"
    objids = [[]]
    if(len(batch_intra_dataFrame.index)>0):
        objids1 = batch_intra_dataFrame['object_id'].values.tolist()
        objids = []
        for id in objids1:
            l1 = [id]
            objids.append(l1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('object_id', axis=1)
        intra_ans = intra_model.predict_proba(batch_intra_dataFrame)
        #print intra_model.classes_
        z = np.zeros((len(intra_ans),9)) # zeros for extra classes and class 99
        intra_ans = np.append(z,intra_ans,axis=1)
        z1 = np.zeros((len(intra_ans),1))
        intra_ans = np.append(intra_ans,z1,axis=1)
        intra_ans = np.append(objids,intra_ans,axis=1)

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
    return arr

def main():

    mode = 1 #0-cv, 1-predict

    print "Reading train data"
    training_set_raw = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')
    training_set_metadata_raw = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')

    #training_set_raw = pd.read_csv('../training_set.csv')
    #training_set_metadata_raw = pd.read_csv('../training_set_metadata.csv')


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
        clf = MLPClassifier(max_iter=5)

        scaler = StandardScaler()
        extragalactic_data = fill_in_hostgal_specz(extragalactic_data)
        extragalactic_data.loc[:, extragalactic_data.columns != 'passband' ] = (scaler.fit_transform(extragalactic_data.loc[:,extragalactic_data.columns != 'passband' ]))
        print cross_val_score(clf, extragalactic_data, extragalactic_targets, cv=10, scoring="neg_log_loss").mean()

        print "Model for intra:"
        clf = MLPClassifier()
        param_grid = {
        'hidden_layer_sizes': range(1,6),
        'batch_size': range(5,106)
        }
        intragalactic_data = fill_in_hostgal_specz(intragalactic_data)
        scaler = StandardScaler()
        intragalactic_data.loc[:,intragalactic_data.columns != 'passband' ] = (scaler.fit_transform(intragalactic_data.loc[:,intragalactic_data.columns != 'passband' ]))
        print cross_val_score(clf, intragalactic_data, intragalactic_targets, cv=10, scoring="neg_log_loss").mean()


    else:
        print "Training"
        clf = MLPClassifier(max_iter=5)

        scaler = StandardScaler()
        intragalactic_data.loc[:,intragalactic_data.columns != 'passband' ] = (scaler.fit_transform(intragalactic_data.loc[:,intragalactic_data.columns != 'passband' ]))

        scaler = StandardScaler()
        extragalactic_data.loc[:, extragalactic_data.columns != 'passband' ] = (scaler.fit_transform(extragalactic_data.loc[:,extragalactic_data.columns != 'passband' ]))


        extra_model = clf
        extra_model.fit(extragalactic_data, extragalactic_targets.values.ravel())
        clf = MLPClassifier()
        intra_model = clf
        intragalactic_data = fill_in_hostgal_specz(intragalactic_data)
        intra_model.fit(intragalactic_data, intragalactic_targets.values.ravel())
        print "Finished training. Starting predictions"

        print "Reading test data"
        test_set_metadata_raw = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')
        filepath = '/modules/cs342/Assignment2/test_set.csv'

        extra_classes = extra_model.classes_
        intra_classes = intra_model.classes_


        extra_ids = []
        intra_ids = []
        extra_ids, intra_ids = splitTestGalaxies(test_set_metadata_raw)
        #print "Extra ids"
        #print extra_ids

        #print "Intra_ids"
        #print intra_ids

        column_names = []
        column_names.append('object_id')
        for classi in extra_classes:
            className = "class_" + str(classi)
            column_names.append(className)
        for classi in intra_classes:
            className = "class_" + str(classi)
            column_names.append(className)
        column_names.append("class_99")

        #print column_names
        count = 0
        batch_no = 0
        batch_extra_dataFrame = pd.DataFrame()
        batch_intra_dataFrame = pd.DataFrame()
        myextrabatchlist = []
        myintrabatchlist = []

        test_set_metadata_raw = fill_in_hostgal_specz(test_set_metadata_raw)


        print " >Starting new batch 0"
        my_extra_data_list = []
        my_intra_data_list = []

        extra_idss = set(extra_ids)
        intra_idss = set(intra_ids)
        cc=-1
        for obj_id, d in get_objects_by_id(filepath):
            cc=cc+1
            #combined = format(test_set_metadata_raw.loc[test_set_metadata_raw['object_id']==obj_id],d)
            if (obj_id in extra_idss):
                my_extra_data_list.append(d) # = np.append(my_extra_data_list, d)
            else:
                my_intra_data_list.append(d) #= np.append(my_intra_data_list, d)
            if(count == 10000):
                print " >>Formatting batch objects"
                arr = my_predict(column_names,my_extra_data_list, my_intra_data_list, test_set_metadata_raw, extra_model, intra_model)

                print " >>Write to csv"
                finish = pd.DataFrame(arr, columns=column_names)
                #print finish['object_id']
                if(batch_no==0):
                    finish.to_csv("preds_mlpraw.csv", index = False, header = True)
                else:
                    with open('preds_mlpraw.csv', 'a') as f:
                        finish.to_csv(f, index = False, header=False)

                print " >Starting new batch " + str(batch_no + 1)
                batch_no = batch_no + 1
                lst = 0
                count = 0
                my_extra_data_list = []
                my_intra_data_list = []

            else:
                count = count + 1
        print "!Remaining objects: " + str(count)
        print " >>Formatting batch objects"
        arr = my_predict(column_names,my_extra_data_list, my_intra_data_list, test_set_metadata_raw, extra_model, intra_model)

        print " >>Write to csv"
        finish = pd.DataFrame(arr, columns=column_names)
        with open('preds_mlpraw.csv', 'a') as f:
            finish.to_csv(f, index = False, header=False)


        print " >>Clean up."
        preds = pd.read_csv('preds_mlpraw.csv')
        preds['object_id']=preds['object_id'].apply(int)
	    #preds['object_id']=preds['object_id'].apply(int)
        print preds.shape
        print cc
        preds.to_csv("preds_mlpraw2.csv", index=False)
	    #preds.to_csv('predictions2.csv', index=False)

        print "DONE."

main()
