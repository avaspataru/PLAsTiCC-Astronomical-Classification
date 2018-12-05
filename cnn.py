import pandas as pd
import numpy as np
import scipy.signal as signal
import operator
import time
import gc
import math
from gatspy import periodic
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn import tree
from sklearn import preprocessing
from random import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool
import multiprocessing as mp
from gatspy.periodic import LombScargleFast
from functools import partial
from keras.models import Sequential, Model

import keras
import tensorflow as tf
import keras.backend as K
from keras import regularizers
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

from keras.wrappers.scikit_learn import KerasClassifier

def format(set_metadata_raw, set_raw):

    arr =[]
    for obj in set_metadata_raw['object_id'].unique():
        df = training_set_raw.loc[training_set_raw['object_id'] == obj].drop('object_id',axis=1)
        filled_matrix = df.as_matrix()
        npad = [(0,352-len(df.index)),(0,0)]
        matrix = np.pad(filled_matrix, pad_width=npad, mode='constant', constant_values=0)
        arr.append(matrix)
    final = np.array(arr)
    print final.shape

    return final

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

def my_predict(column_names,arr_matrix_extra, arr_matrix_intra, test_set_metadata_raw, extra_model, intra_model):

    X_extra = np.array(arr_matrix_extra)
    X_intra = np.array(arr_matrix_intra)

    extra_ans = []
    intra_ans = []

    print " >>Predicting extra"
    extra_ans = extra_model.predict_proba(X_extra)
    z = np.zeros((len(extra_ans),6)) # zeros for intra classes and class 99
    extra_ans = np.append(extra_ans,z,axis=1)

    print " >>Predicting intra"
    intra_ans = intra_model.predict_proba(X_intra)

    #print intra_model.classes_
    z = np.zeros((len(intra_ans),9)) # zeros for extra classes and class 99
    intra_ans = np.append(z,intra_ans,axis=1)
    z1 = np.zeros((len(intra_ans),1))
    intra_ans = np.append(intra_ans,z1,axis=1)

    print " >>Putting together"
    arr = []
    arr = np.concatenate((extra_ans,intra_ans), axis=0)
    return arr

def augument(data, meta):
    gc.enable()
    print "Augumenting training set data"
    print ">>compute ids"
    idmax = meta['object_id'].max()
    n = len(meta.index)
    m = len(data.index)
    oldids = meta['object_id'].unique()
    newids = np.array(range(idmax+1, idmax+n+1))

    print ">>change meta"
    new_meta = meta.copy()
    new_meta['object_id'] = newids

    #add noise to distmod
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, [1,n])[0]
    new_meta['distmod'] = new_meta['distmod'] + noise
    final_meta = meta.append(new_meta)

    print ">>change data"
    new_data = data.copy()
    gc.collect()
    dictionary = dict(zip(oldids, newids))
    new_data = new_data.replace({"object_id": dictionary})
    #print new_data
    #add noise to flux

    fluxerrvals = new_data['flux_err'].apply(abs).values.tolist()

    noise = np.random.normal(mu, new_data['flux_err'])[0]
    new_data['flux'] = new_data['flux'] + noise
    #add noise to flux_err
    noise = np.random.normal(mu, sigma, [1,m])[0]
    new_data['flux_err'] = new_data['flux_err'] + noise

    final_data = data.append(new_data)
    #print final_meta
    print ">>finished augumenting."
    print len(final_data['object_id'].unique())
    print len(final_meta['object_id'].unique())
    final_meta.to_csv('train_meta_aug4.csv', index=False)
    final_data.to_csv('train_data_aug4.csv', index=False)
    return final_data, final_meta

def augument_twice(data, meta):
    gc.enable()
    print ">>compute ids"
    idmax = meta['object_id'].max()
    n = len(meta.index)
    m = len(data.index)
    oldids = meta['object_id'].unique()
    newids = np.array(range(idmax+1, idmax+n+1))
    newids2 = np.array(range(idmax+n+2, idmax+n+2+n))

    print ">>change meta"
    new_meta = meta.copy()
    new_meta2 = meta.copy()
    new_meta['object_id'] = newids
    new_meta2['object_id'] = newids2

    #add noise to distmod
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, [1,n])[0]
    noise2 = np.random.normal(mu, sigma, [1,n])[0]
    new_meta['distmod'] = new_meta['distmod'] + noise
    new_meta2['distmod'] = new_meta2['distmod'] + noise2
    final_meta = meta.append(new_meta)
    final_meta = final_meta.append(new_meta2)

    print ">>change data"
    new_data = data.copy()
    new_data2 = data.copy()
    gc.collect()
    dictionary = dict(zip(oldids, newids))
    dictionary2 = dict(zip(oldids, newids2))
    new_data = new_data.replace({"object_id": dictionary})
    new_data2 = new_data2.replace({"object_id": dictionary2})
    #print new_data
    #add noise to flux

    noise = np.random.normal(mu, new_data['flux_err'])[0]
    noise2 = np.random.normal(mu, new_data2['flux_err'])[0]
    new_data['flux'] = new_data['flux'] + noise
    new_data2['flux'] = new_data2['flux'] + noise2
    #add noise to flux_err
    noise = np.random.normal(mu, sigma, [1,m])[0]
    noise2 = np.random.normal(mu, sigma, [1,m])[0]
    new_data['flux_err'] = new_data['flux_err'] + noise
    new_data2['flux_err'] = new_data2['flux_err'] + noise2

    final_data = data.append(new_data)
    final_data = final_data.append(new_data2)
    #print final_meta
    print ">>finished augumenting."
    print len(final_data['object_id'].unique())
    print len(final_meta['object_id'].unique())
    final_meta.to_csv('train_meta_aug.csv', index=False)
    final_data.to_csv('train_data_aug.csv', index=False)
    return final_data, final_meta

# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true,y_pred):
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss

def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

K.clear_session()
def build_model():
    model = Sequential()
    model.add(Conv1D(32,6,activation='relu', input_shape=(352,6)))

    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D())
    #model.add(Dropout(0.1))
    #model.add(Flatten())
    model.add(Conv1D(32,6,activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D())
    model.add(Flatten())
    #model.add(Dense(14, activation='softmax'))
    model.add(Dense(14, activation='softmax'))
    return model


def load_dmdt_images(objects, base_dir='train'):
    dmdt_img_dict = OrderedDict()
    for obj in objects:
        key = '{}/{}_dmdt.pkl'.format(base_dir, obj)
        if os.path.isfile(key):
            with(open(key, 'rb')) as f:
                dmdt_img_dict[obj] = pickle.load(f)
    return dmdt_img_dict

def build_model_extra():
    model = Sequential()
    model.add(Conv1D(64,12,activation='relu', input_shape=(352,5)))

    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D())
    #model.add(Dropout(0.1))
    #model.add(Flatten())
    model.add(Conv1D(64,12,activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D())
    model.add(Flatten())
    #model.add(Dense(14, activation='softmax'))
    model.add(Dense(9, activation='softmax'))
    model.compile(loss='categorical_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric
    return model

def build_model_intra():
    model = Sequential()
    model.add(Conv1D(64,12,activation='relu', input_shape=(352,5)))

    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D())
    #model.add(Dropout(0.1))
    #model.add(Flatten())
    model.add(Conv1D(64,12,activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D())
    model.add(Flatten())
    #model.add(Dense(14, activation='softmax'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric
    return model

def normalizeClasses(values):
    classes = np.unique(values)
    newcls = []
    ind = 0
    for cl in classes:
        newcls.append(ind)
        ind = ind+1
    dictionary = dict(zip(classes, newcls))
    print dictionary
    old_classes = values
    new_classes = [dictionary[letter] for letter in old_classes]
    return old_classes, new_classes, dictionary

def main():

    mode = 1 #0-cv, 1-predict

    print "Reading train data"
    training_set_raw = pd.read_csv('./train_data_aug4.csv')
    training_set_metadata_raw = pd.read_csv('./train_meta_aug4.csv')

    #print training_set_metadata_raw['hostgal_photoz']

    extra_ids = training_set_metadata_raw.loc[training_set_metadata_raw['hostgal_photoz'] != 0]['object_id']
    intra_ids = training_set_metadata_raw.loc[training_set_metadata_raw['hostgal_photoz'] == 0]['object_id']
    #print intra_ids

    training_set_targets_extra = training_set_metadata_raw.loc[training_set_metadata_raw['hostgal_photoz'] != 0]['target']
    training_set_targets_intra = training_set_metadata_raw.loc[training_set_metadata_raw['hostgal_photoz'] == 0]['target']

    old_extra_cl, new_extra_cl, extra_dict = normalizeClasses(training_set_targets_extra.values)
    old_intra_cl, new_intra_cl, intra_dict = normalizeClasses(training_set_targets_intra.values)

    column_names = []
    #column_names.append('object_id')
    for classi in training_set_targets_extra.unique():
        className = "class_" + str(classi)
        column_names.append(className)
    for classi in training_set_targets_intra.unique():
        className = "class_" + str(classi)
        column_names.append(className)
    column_names.append("class_99")

    print column_names

    #extra galactic
    print "Training extra galactic cnn"
    print ">>format"
    arr =[]
    for obj in extra_ids:
        df = training_set_raw.loc[training_set_raw['object_id'] == obj].drop('object_id',axis=1)
        filled_matrix = df.as_matrix()
        npad = [(0,352-len(df.index)),(0,0)]
        matrix = np.pad(filled_matrix, pad_width=npad, mode='constant', constant_values=0)
        arr.append(matrix)
    final = np.array(arr)
    print final.shape

    y = np.array(pd.get_dummies(training_set_targets_extra))
    print ">>finished format"
    extra_model = KerasClassifier(build_fn =build_model_extra, verbose=1)
    print "built"
    print final.shape #(7848, 352, 6)
    print y.shape #(7848, 14)
    extra_model.fit(final,y)

    #intra galactic
    print "Training intragalactic cnn"
    arr =[]
    for obj in intra_ids:
        df = training_set_raw.loc[training_set_raw['object_id'] == obj].drop('object_id',axis=1)
        filled_matrix = df.as_matrix()
        npad = [(0,352-len(df.index)),(0,0)]
        matrix = np.pad(filled_matrix, pad_width=npad, mode='constant', constant_values=0)
        arr.append(matrix)
    final = np.array(arr)
    print final.shape

    y = np.array(pd.get_dummies(training_set_targets_intra))

    intra_model = KerasClassifier(build_fn =build_model_intra, verbose=1)
    print "built"

    print final.shape #(7848, 352, 6)
    print y.shape #(7848, 14)
    intra_model.fit(final,y)

    if mode==1:
        print "Finished training. Starting predictions"

        print "Reading test data"
        test_set_metadata_raw = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')
        filepath = '/modules/cs342/Assignment2/test_set.csv'

        extra_ids = test_set_metadata_raw.loc[test_set_metadata_raw['hostgal_photoz'] != 0]['object_id']
        intra_ids = test_set_metadata_raw.loc[test_set_metadata_raw['hostgal_photoz'] == 0]['object_id']

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
        arr_matrix_extra = []
        arr_matrix_intra = []

        arr_ids_extra = []
        arr_ids_intra = []
        for obj_id, d in get_objects_by_id(filepath):
            cc=cc+1
            #combined = format(test_set_metadata_raw.loc[test_set_metadata_raw['object_id']==obj_id],d)
            if (obj_id in extra_idss):
                d = d.drop('object_id',axis=1)
                filled_matrix = d.as_matrix()
                npad = [(0,352-len(d.index)),(0,0)]
                matrix = np.pad(filled_matrix, pad_width=npad, mode='constant', constant_values=0)
                arr_matrix_extra.append(matrix)
                arr_ids_extra.append(obj_id)
            else:
                d = d.drop('object_id',axis=1)
                filled_matrix = d.as_matrix()
                npad = [(0,352-len(d.index)),(0,0)]
                matrix = np.pad(filled_matrix, pad_width=npad, mode='constant', constant_values=0)
                arr_matrix_intra.append(matrix)
                arr_ids_intra.append(obj_id)
            if(count == 10000):
                print " >>Formatting batch objects"
                arr = my_predict(column_names,arr_matrix_extra, arr_matrix_intra, test_set_metadata_raw, extra_model, intra_model)

                print " >>Write to csv"
                finish = pd.DataFrame(arr, columns=column_names)
                finish["object_id"] = np.concatenate((arr_ids_extra,arr_ids_intra),axis=0)
                finish["class_99"] = (1-finish.drop("object_id", axis=1)).product(axis=1) #Adding values to class_99
                #Below is a very messy way of making all rows sum to 1 despite the above
                finish.loc[:,finish.columns!="object_id"] = finish.loc[:,finish.columns!="object_id"].div(finish.loc[:,finish.columns!="object_id"].sum(axis=1), axis=0)
                if(batch_no==0):
                    finish.to_csv("predictionsCNN.csv", index = False, header = True)
                else:
                    with open('predictionsCNN.csv', 'a') as f:
                        finish.to_csv(f, index = False, header=False)

                print " >Starting new batch " + str(batch_no + 1)
                batch_no = batch_no + 1
                lst = 0
                count = 0
                arr_matrix_extra = []
                arr_matrix_intra = []

                arr_ids_extra = []
                arr_ids_intra = []

            else:
                count = count + 1
        print "!Remaining objects: " + str(count)
        print " >>Formatting batch objects"
        arr = my_predict(column_names,arr_matrix_extra, arr_matrix_intra, test_set_metadata_raw, extra_model, intra_model)

        print " >>Write to csv"
        finish = pd.DataFrame(arr, columns=column_names)
        finish["object_id"] = np.concatenate((arr_ids_extra,arr_ids_intra),axis=0)
        finish["class_99"] = (1-finish.drop("object_id", axis=1)).product(axis=1) #Adding values to class_99
        #Below is a very messy way of making all rows sum to 1 despite the above
        finish.loc[:,finish.columns!="object_id"] = finish.loc[:,finish.columns!="object_id"].div(finish.loc[:,finish.columns!="object_id"].sum(axis=1), axis=0)
        if(batch_no==0):
            finish.to_csv("predictionsCNN.csv", index = False, header = True)
        else:
            with open('predictionsCNN.csv', 'a') as f:
                finish.to_csv(f, index = False, header=False)

        print " >>Clean up."
        preds = pd.read_csv('predictionsCNN.csv')
        preds['object_id']=preds['object_id'].apply(int)
	    #preds['object_id']=preds['object_id'].apply(int)
        print preds.shape
        print cc
        preds.to_csv('predictionsCNN2.csv', index=False)
	    #preds.to_csv('predictions2.csv', index=False)

        print "DONE."

main()
