import pandas as pd
import numpy as np
import scipy.signal as signal
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
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool
import multiprocessing as mp
from gatspy.periodic import LombScargleFast

CORES = mp.cpu_count() #4

def scale(dataFrame):
    df = dataFrame.copy()
    col = df.columns.values

    for c in col:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return dataFrame

def splitGalaxies(dataFrame, targets):
    print "Split extragalactic "
    extra = np.where(dataFrame['hostgal_specz']==0.0)
    extragalactic_data = dataFrame.drop(dataFrame.index[extra])
    extra_ids = extragalactic_data['object_id'].values.tolist()
    #extragalactic_data = extragalactic_data.drop('object_id',axis=1)
    extragalactic_targets = targets.drop(targets.index[extra])


    print "Split intragalactic "
    intra = np.where(dataFrame['hostgal_specz']!=0.0)
    intragalactic_data = dataFrame.drop(dataFrame.index[intra])
    intra_ids = intragalactic_data['object_id'].values.tolist()
    #intragalactic_data = intragalactic_data.drop('object_id',axis=1)
    intragalactic_targets = targets.drop(targets.index[intra])

    return extragalactic_data, extragalactic_targets,extra_ids, intragalactic_data, intragalactic_targets, intra_ids

def splitTestGalaxies(dataFrame):
    print "Split extragalactic "
    extra = np.where(dataFrame['hostgal_specz']==0.0)
    extragalactic_data = dataFrame.drop(dataFrame.index[extra])
    extra_ids = extragalactic_data['object_id'].values.tolist()

    print "Split intragalactic "
    intra = np.where(dataFrame['hostgal_specz']!=0.0)
    intragalactic_data = dataFrame.drop(dataFrame.index[intra])
    intra_ids = intragalactic_data['object_id'].values.tolist()

    return extra_ids,intra_ids

def format(set_metadata_raw, set_raw):

    print "BEGIN FORMAT -----"
    #set_data = set_metadata_raw.drop('distmod',axis=1)
    #set_raw['flux'] = set_raw['flux'] * set_data['mwebv']

    set_data = set_metadata_raw.drop('mwebv', axis=1)

    set_raw['flux_ratio_sq'] = np.power(set_raw['flux'] / set_raw['flux_err'], 2.0)
    set_raw['flux_by_flux_ratio_sq'] = set_raw['flux'] * set_raw['flux_ratio_sq']

    aggs = {
    'flux': ['min', 'max', 'mean'],
    'detected': ['max'],
    'flux_ratio_sq':['sum'],
    'flux_by_flux_ratio_sq':['mean']
    }

    agg_train = set_raw.groupby(['object_id','passband']).agg(aggs).reset_index()

    agg_train.columns = [name[0]+"_"+name[1] for name in agg_train.columns]

    agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
    agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
    agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_mean'] / agg_train['flux_ratio_sq_sum']
    agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']

    agg_train.head()

    del set_raw
    gc.collect()

    full_train = agg_train

    full_train['magn'] = -2.5*full_train["flux_mean"].apply(np.log)
    #print full_train.columns
    #merge the pass bands
    cc = []
    for coln in full_train.columns:
        if(coln == 'object_id_'):
            cc.append('object_id')
        else:
            c = coln + '_' + str(0)
            cc.append(c)

    p0 = full_train.loc[full_train['passband_'] == 0]
    p0df = pd.DataFrame(p0.values,columns=cc)
    p0df = p0df.drop('passband__0',axis=1)

    cc = [
    coln + '_' + str(1) for coln in full_train.columns
    ]
    p1 = full_train.loc[full_train['passband_'] == 1]
    p1df = pd.DataFrame(p1.values,columns=cc)
    p1df = p1df.drop('object_id__1',axis=1)
    p1df = p1df.drop('passband__1',axis=1)

    cc = [
    coln + '_' + str(2) for coln in full_train.columns
    ]
    p2 = full_train.loc[full_train['passband_'] == 2]
    p2df = pd.DataFrame(p2.values,columns=cc)
    p2df = p2df.drop('object_id__2',axis=1)
    p2df = p2df.drop('passband__2',axis=1)

    cc = [
    coln + '_' + str(3) for coln in full_train.columns
    ]
    p3 = full_train.loc[full_train['passband_'] == 3]
    p3df = pd.DataFrame(p3.values,columns=cc)
    p3df = p3df.drop('object_id__3',axis=1)
    p3df = p3df.drop('passband__3',axis=1)

    cc = [
    coln + '_' + str(4) for coln in full_train.columns
    ]
    p4 = full_train.loc[full_train['passband_'] == 4]
    p4df = pd.DataFrame(p4.values,columns=cc)
    p4df = p4df.drop('object_id__4',axis=1)
    p4df = p4df.drop('passband__4',axis=1)

    cc = [
    coln + '_' + str(5) for coln in full_train.columns
    ]
    p5 = full_train.loc[full_train['passband_'] == 5]
    p5df = pd.DataFrame(p5.values,columns=cc)
    p5df = p5df.drop('object_id__5',axis=1)
    p5df = p5df.drop('passband__5',axis=1)

    tog = pd.concat([p0df,p1df],axis=1)
    tog = pd.concat([tog,p2df],axis =1)
    tog = pd.concat([tog,p3df],axis =1)
    tog = pd.concat([tog,p4df],axis =1)
    tog = pd.concat([tog,p5df],axis =1)

    #print new_columns
    full_train = tog

    full_train = full_train.reset_index().merge(
        right=set_data,
        how='outer',
        on='object_id'
        )
    full_train= full_train.drop('index',axis=1)

    full_train.loc[full_train['magn_0'].isnull(),'magn_0'] = 0 #abs mag should be 0 if no flux ?
    full_train.loc[full_train['magn_1'].isnull(),'magn_1'] = 0
    full_train.loc[full_train['magn_2'].isnull(),'magn_2'] = 0
    full_train.loc[full_train['magn_3'].isnull(),'magn_3'] = 0
    full_train.loc[full_train['magn_4'].isnull(),'magn_4'] = 0
    full_train.loc[full_train['magn_5'].isnull(),'magn_5'] = 0
    full_train['absmagn_0'] = full_train['magn_0'] - full_train['distmod']
    full_train['absmagn_1'] = full_train['magn_1'] - full_train['distmod']
    full_train['absmagn_2'] = full_train['magn_2'] - full_train['distmod']
    full_train['absmagn_3'] = full_train['magn_3'] - full_train['distmod']
    full_train['absmagn_4'] = full_train['magn_4'] - full_train['distmod']
    full_train['absmagn_5'] = full_train['magn_5'] - full_train['distmod']

    full_train.loc[full_train['magn_0'] == 0,'absmagn_0'] = 0 #abs mag should be 0 if no flux ?
    full_train.loc[full_train['magn_1'] == 0,'absmagn_1'] = 0
    full_train.loc[full_train['magn_2'] == 0,'absmagn_2'] = 0
    full_train.loc[full_train['magn_3'] == 0,'absmagn_3'] = 0
    full_train.loc[full_train['magn_4'] == 0,'absmagn_4'] = 0
    full_train.loc[full_train['magn_5'] == 0,'absmagn_5'] = 0



    full_train = full_train.drop('distmod',axis=1)
    #print full_train.columns

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

def do_periods(set_raw):
    unqobjid = set_raw['object_id'].unique()
    cou = 0
    ccols = ['object_id','period_0','period_1','period_2','period_3','period_4','period_5']
    periods_list = []
    for ob in unqobjid:
        if cou%500 == 0:
            print "COUNT " + str(cou)
        thisobjper = [ob]
        for passb in range(0,6):
            obdf = set_raw.loc[set_raw['object_id'] == ob]
            obdf = obdf.loc[obdf['passband'] == passb]

            x = np.array(obdf['mjd'].values.tolist())
            y = np.array(obdf['flux'].values.tolist())
            #model = LombScargleFast(fit_period=True)
            #t_min = max(np.median(np.diff(sorted(obdf['mjd']))), 0.1)
            #t_max = min(10., (obdf['mjd'].max() - obdf['mjd'].min())/2.)
            f = np.array([0.5,33,0.5])
            pgram = signal.lombscargle(x, y, f)
            period = np.max(pgram)
            #print "Period " + str(period)
            #model.optimizer.set(period_range=(t_min, t_max))
            #model.fit(obdf['mjd'], obdf['flux'], dy=obdf['flux_err'])
            #period = model.best_period
            thisobjper.append(period)
        cou = cou + 1
        periods_list.append(thisobjper)

    periods = pd.DataFrame(periods_list,columns=ccols)
    periods.to_csv('periods_train.csv', index=False)
    return periods

def fill_in_hostgal_specz(dataFrame):
    df = dataFrame.copy()

    df.loc[df['hostgal_specz'].isnull(),'hostgal_specz'] = df['hostgal_photoz']

    df = df.drop('hostgal_photoz',axis=1)
    df = df.drop('hostgal_photoz_err',axis=1)
    #df = df.drop('distmod',axis=1) already dropped
    df = df.drop('ra',axis=1)
    df = df.drop('decl',axis=1)
    df = df.drop('gal_l',axis=1)
    df = df.drop('gal_b',axis=1)
    return df

def my_predict(column_names,my_extra_data_list, my_intra_data_list, test_set_metadata_raw, extra_model, intra_model):

    formatted_columns = [u'object_id', u'flux_min_0', u'flux_max_0',
       u'flux_mean_0', u'flux_std_0', u'detected_max_0',
       u'flux_by_flux_ratio_sq_mean_0', u'flux_ratio_sq_sum_0',
       u'flux_diff_0', u'flux_dif2_0', u'flux_w_mean_0', u'flux_dif3_0',
       u'flux_min_1', u'flux_max_1', u'flux_mean_1',
       u'flux_std_1', u'detected_max_1',
       u'flux_by_flux_ratio_sq_mean_1', u'flux_ratio_sq_sum_1',
       u'flux_diff_1', u'flux_dif2_1', u'flux_w_mean_1', u'flux_dif3_1',
       u'flux_min_2', u'flux_max_2', u'flux_mean_2',
       u'flux_std_2', u'detected_max_2',
       u'flux_by_flux_ratio_sq_mean_2', u'flux_ratio_sq_sum_2',
       u'flux_diff_2', u'flux_dif2_2', u'flux_w_mean_2', u'flux_dif3_2',
       u'flux_min_3', u'flux_max_3', u'flux_mean_3',
       u'flux_std_3', u'detected_max_3',
       u'flux_by_flux_ratio_sq_mean_3', u'flux_ratio_sq_sum_3',
       u'flux_diff_3', u'flux_dif2_3', u'flux_w_mean_3', u'flux_dif3_3',
       u'flux_min_4', u'flux_max_4', u'flux_mean_4',
       u'flux_std_4', u'detected_max_4',
       u'flux_by_flux_ratio_sq_mean_4', u'flux_ratio_sq_sum_4',
       u'flux_diff_4', u'flux_dif2_4', u'flux_w_mean_4', u'flux_dif3_4',
       u'flux_min_5', u'flux_max_5', u'flux_mean_5',
       u'flux_std_5', u'detected_max_5',
       u'flux_by_flux_ratio_sq_mean_5', u'flux_ratio_sq_sum_5',
       u'flux_diff_5', u'flux_dif2_5', u'flux_w_mean_5', u'flux_dif3_5',
       u'ddf', u'hostgal_specz', u'period_0', u'period_1', u'period_2', u'period_3',
       u'period_4', u'period_5', u'magn_0', u'magn_1', u'magn_2', u'magn_3',
       u'magn_4', u'magn_5', u'absmagn_0', u'absmagn_1', u'absmagn_2', u'absmagn_3',
       u'absmagn_4', u'absmagn_5']


    finish = pd.DataFrame(columns=column_names)

    batch_extra_dataFrame = pd.DataFrame(columns = formatted_columns)
    batch_intra_dataFrame = pd.DataFrame(columns = formatted_columns)
    my_extra_data_batch = pd.DataFrame(columns = ['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected'])
    my_intra_data_batch = pd.DataFrame(columns = ['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected'])


    if(len(my_extra_data_list)>0):
        my_extra_data_batch = pd.concat(my_extra_data_list)
    if(len(my_intra_data_list)>0):
        my_intra_data_batch = pd.concat(my_intra_data_list)

    initial_intra = my_intra_data_batch
    intra_periods = do_periods(initial_intra)
    intra_periods['period_0'] = (intra_periods['period_0']-intra_periods['period_0'].mean()) / intra_periods['period_0'].std()
    intra_periods['period_1'] = (intra_periods['period_1']-intra_periods['period_1'].mean()) / intra_periods['period_1'].std()
    intra_periods['period_2'] = (intra_periods['period_2']-intra_periods['period_2'].mean()) / intra_periods['period_2'].std()
    intra_periods['period_3'] = (intra_periods['period_3']-intra_periods['period_3'].mean()) / intra_periods['period_3'].std()
    intra_periods['period_4'] = (intra_periods['period_4']-intra_periods['period_4'].mean()) / intra_periods['period_4'].std()
    intra_periods['period_5'] = (intra_periods['period_5']-intra_periods['period_5'].mean()) / intra_periods['period_5'].std()

    tt1 = test_set_metadata_raw.loc[test_set_metadata_raw['object_id'].isin(my_extra_data_batch['object_id'].values.tolist())]
    if(len(my_extra_data_batch.index)>0):
        batch_extra_dataFrame= format(tt1, my_extra_data_batch)
    else:
        batch_extra_dataFrame = pd.DataFrame(columns = formatted_columns)
    tt2 = test_set_metadata_raw.loc[test_set_metadata_raw['object_id'].isin(my_intra_data_batch['object_id'].values.tolist())]
    if(len(my_intra_data_batch.index)>0):
        batch_intra_dataFrame= format(tt2, my_intra_data_batch)
    else:
        batch_intra_dataFrame = pd.DataFrame(columns = formatted_columns)

    print "NULLS"
    print batch_extra_dataFrame.columns[batch_extra_dataFrame.isnull().any()].tolist()
    print batch_intra_dataFrame.columns[batch_intra_dataFrame.isnull().any()].tolist()

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
        batch_intra_dataFrame = batch_intra_dataFrame.merge(
            right=intra_periods,
            how='outer',
            on='object_id'
            )
        batch_intra_dataFrame = batch_intra_dataFrame.drop('magn_0', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('magn_1', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('magn_2', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('magn_3', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('magn_4', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('magn_5', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('absmagn_0', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('absmagn_1', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('absmagn_2', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('absmagn_3', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('absmagn_4', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('absmagn_5', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_diff_0', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_diff_1', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_diff_2', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_diff_3', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_diff_4', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_diff_5', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif2_0', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif2_1', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif2_2', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif2_3', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif2_4', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif2_5', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif3_0', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif3_1', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif3_2', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif3_3', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif3_4', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_dif3_5', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_w_mean_0', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_w_mean_1', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_w_mean_2', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_w_mean_3', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_w_mean_4', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_w_mean_5', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_ratio_sq_sum_0', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_ratio_sq_sum_1', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_ratio_sq_sum_2', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_ratio_sq_sum_3', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_ratio_sq_sum_4', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_ratio_sq_sum_5', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_by_flux_ratio_sq_mean_0', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_by_flux_ratio_sq_mean_1', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_by_flux_ratio_sq_mean_2', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_by_flux_ratio_sq_mean_3', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_by_flux_ratio_sq_mean_4', axis = 1)
        batch_intra_dataFrame = batch_intra_dataFrame.drop('flux_by_flux_ratio_sq_mean_5', axis = 1)
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

    #classes, not class 99
    training_set_targets = training_set_metadata_raw['target']
    training_set_data = training_set_metadata_raw.drop('target',axis=1)

    classes = sorted(training_set_targets.unique())
    class_weight = {
    c: 1 for c in classes
    }
    for c in [64, 15]:
        class_weight[c] = 2

    training_set_data = fill_in_hostgal_specz(training_set_data)
    full_train = format(training_set_data, training_set_raw)
    extragalactic_data, extragalactic_targets, extra_ids,  intragalactic_data, intragalactic_targets, intra_ids = splitGalaxies(full_train, training_set_targets)

    initial_intra = training_set_raw.loc[training_set_raw['object_id'].isin(intra_ids)]
    intra_periods = do_periods(initial_intra)
    #intra_periods = pd.read_csv('./periods_train.csv')
    intra_periods['period_0'] = (intra_periods['period_0']-intra_periods['period_0'].mean()) / intra_periods['period_0'].std()
    intra_periods['period_1'] = (intra_periods['period_1']-intra_periods['period_1'].mean()) / intra_periods['period_1'].std()
    intra_periods['period_2'] = (intra_periods['period_2']-intra_periods['period_2'].mean()) / intra_periods['period_2'].std()
    intra_periods['period_3'] = (intra_periods['period_3']-intra_periods['period_3'].mean()) / intra_periods['period_3'].std()
    intra_periods['period_4'] = (intra_periods['period_4']-intra_periods['period_4'].mean()) / intra_periods['period_4'].std()
    intra_periods['period_5'] = (intra_periods['period_5']-intra_periods['period_5'].mean()) / intra_periods['period_5'].std()

    initial_extra = training_set_raw.loc[training_set_raw['object_id'].isin(extra_ids)]

    intragalactic_data = intragalactic_data.merge(
        right=intra_periods,
        how='outer',
        on='object_id'
        )
    #print intragalactic_data
    intragalactic_data = intragalactic_data.drop('object_id',axis=1)
    intragalactic_data = intragalactic_data.drop('magn_0', axis = 1)
    intragalactic_data = intragalactic_data.drop('magn_1', axis = 1)
    intragalactic_data = intragalactic_data.drop('magn_2', axis = 1)
    intragalactic_data = intragalactic_data.drop('magn_3', axis = 1)
    intragalactic_data = intragalactic_data.drop('magn_4', axis = 1)
    intragalactic_data = intragalactic_data.drop('magn_5', axis = 1)
    intragalactic_data = intragalactic_data.drop('absmagn_0', axis = 1)
    intragalactic_data = intragalactic_data.drop('absmagn_1', axis = 1)
    intragalactic_data = intragalactic_data.drop('absmagn_2', axis = 1)
    intragalactic_data = intragalactic_data.drop('absmagn_3', axis = 1)
    intragalactic_data = intragalactic_data.drop('absmagn_4', axis = 1)
    intragalactic_data = intragalactic_data.drop('absmagn_5', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_diff_0', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_diff_1', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_diff_2', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_diff_3', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_diff_4', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_diff_5', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif2_0', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif2_1', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif2_2', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif2_3', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif2_4', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif2_5', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif3_0', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif3_1', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif3_2', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif3_3', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif3_4', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_dif3_5', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_w_mean_0', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_w_mean_1', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_w_mean_2', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_w_mean_3', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_w_mean_4', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_w_mean_5', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_ratio_sq_sum_0', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_ratio_sq_sum_1', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_ratio_sq_sum_2', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_ratio_sq_sum_3', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_ratio_sq_sum_4', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_ratio_sq_sum_5', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_by_flux_ratio_sq_mean_0', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_by_flux_ratio_sq_mean_1', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_by_flux_ratio_sq_mean_2', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_by_flux_ratio_sq_mean_3', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_by_flux_ratio_sq_mean_4', axis = 1)
    intragalactic_data = intragalactic_data.drop('flux_by_flux_ratio_sq_mean_5', axis = 1)

    extragalactic_data = extragalactic_data.drop('object_id',axis=1)

    if mode==0:
        print "Model for extra:"
        param_grid = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
        }
        clf = RandomForestClassifier(n_jobs=2, max_depth=20,n_estimators=100)
        #CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
        #CV_rfc.fit(extragalactic_data, extragalactic_targets)
        #print "params"
        #print CV_rfc.best_params_
        print cross_val_score(clf, extragalactic_data, extragalactic_targets, cv=10, scoring="neg_log_loss").mean()

        print "Model for intra:"
        clf = RandomForestClassifier(n_jobs=2, max_depth=15,n_estimators=100)
        print cross_val_score(clf, intragalactic_data, intragalactic_targets, cv=10, scoring="neg_log_loss").mean()

    else:
        print "Training"
        extra_model = RandomForestClassifier(n_jobs=2, max_depth=20,n_estimators=100)
        extra_model.fit(extragalactic_data, extragalactic_targets.values.ravel())
        intra_model = RandomForestClassifier(n_jobs=2, max_depth=20,n_estimators=100)
        intra_model.fit(intragalactic_data, intragalactic_targets.values.ravel())
        print "Finished training. Starting predictions"

        print "Reading test data"
        test_set_metadata_raw = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')
        filepath = '/modules/cs342/Assignment2/test_set.csv'

        extra_classes = extra_model.classes_
        intra_classes = intra_model.classes_

        extra_ids = []
        intra_ids = []
        test_set_metadata_raw = fill_in_hostgal_specz(test_set_metadata_raw)
        extra_ids, intra_ids = splitTestGalaxies(test_set_metadata_raw)


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
                finish["class_99"] = (1-finish.drop("object_id", axis=1)).product(axis=1) #Adding values to class_99
                #Below is a very messy way of making all rows sum to 1 despite the above
                finish.loc[:,finish.columns!="object_id"] = finish.loc[:,finish.columns!="object_id"].div(finish.loc[:,finish.columns!="object_id"].sum(axis=1), axis=0)

                if(batch_no==0):
                    finish.to_csv("predictions.csv", index = False, header = True)
                else:
                    with open('predictions.csv', 'a') as f:
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
        finish["class_99"] = (1-finish.drop("object_id", axis=1)).product(axis=1) #Adding values to class_99
        #Below is a very messy way of making all rows sum to 1 despite the above
        finish.loc[:,finish.columns!="object_id"] = finish.loc[:,finish.columns!="object_id"].div(finish.loc[:,finish.columns!="object_id"].sum(axis=1), axis=0)
        with open('predictions.csv', 'a') as f:
            finish.to_csv(f, index = False, header=False)


        print " >>Clean up."
        preds = pd.read_csv('predictions.csv')
        preds['object_id']=preds['object_id'].apply(int)
	    #preds['object_id']=preds['object_id'].apply(int)
        print preds.shape
        print cc
        preds.to_csv('predictions2.csv', index=False)
	    #preds.to_csv('predictions2.csv', index=False)

        print "DONE."

main()
