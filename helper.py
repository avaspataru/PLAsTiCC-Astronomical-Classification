import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def my_merge(dataFrame1,dataFrame2):
    df1 = dataFrame1.copy()
    df2 = dataFrame2.copy()

    df1 = pd.concat([df1,df2], axis=1)
    return df1


def removeFeatures(dataFrame):
    df = dataFrame.copy()

    return df

def standardizeData(dataFrame):
    df = dataFrame.copy()
    col = df.columns.values

    for c in col:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df

def normalizeData(dataFrame):
    df = dataFrame.copy()
    col = df.columns.values

    df = (df - df.mean()) / (df.max() - df.min())

    return df

def equalProbabilities(dataFrame, answerFrame):
    df = dataFrame.copy()
    af = answerFrame.copy()

    #set of all those of class1
    #set1 = af.loc[af['class'] == 1]

    #eliminate 104 of class1, to have equal number of class0 and class1 samples
    #to_eliminate = np.random.choice(set1.index.values, 104)
    #to_keep = list( set(df.index.values) - set(to_eliminate))
    #df = df.drop(df.index[to_eliminate])
    #af = af.drop(af.index[to_eliminate])

    return df,af
