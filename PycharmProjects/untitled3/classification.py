import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE


def encode(dic, df, column_name):
    df.loc[:, df.columns == column_name] = df[[column_name]] \
        .applymap(lambda z: dic[z])
    return df


def load_files():
    train = pd.read_csv("train.csv")
    # train = train[['distanceKM', 'taxiDurationMin',
    #             'vehicleType', 'vehicleOption', 'weight']]
    train_y = train['price']
    train = train.drop(['ID', 'vehicleType','SourceState', 'destinationState', 'price', 'date'], axis=1)

    vehicleType = {'treili': 0, 'khavar': 1, 'joft': 2, 'tak': 3}
    vehicleOption = {'kafi': 0, 'mosaghaf_felezi': 1, 'kompressi': 2, 'bari': 3,
                     'labehdar': 4, 'yakhchali': 5, 'hichkodam': 6, 'mosaghaf_chadori': 7,
                     'transit_chadori': 8}

    # train = encode(vehicleType, train, 'vehicleType')
    train = encode(vehicleOption, train, 'vehicleOption')

    # fill nan
    for column in train.columns:
        train[column] = train[column].fillna(np.mean(train[column]))

    # train['weight'] * train['weight']

    for column, i in zip(train.columns, range(len(train.columns))):
        p = train[column] * train[column]
        train[column + "^2"] = p
    print(train.columns)

    model = linear_model.LinearRegression()
    rfe = RFE(model, 1)
    fit = rfe.fit(train, train_y)
    print(fit.n_features_)
    print(fit.support_)
    print(fit.ranking_)

    # new_train = mutual_info_regression(train, train_y)
    # print(new_train[0,:], train.loc[0,:])
    # print(new_train.shape)
    # # normalize data
    # train = train.loc[:, ['distanceKM', 'taxiDurationMin',
    #                       'vehicleType', 'vehicleOption', 'weight']]
    # # print(x)
    # train = StandardScaler().fit_transform(train)
    # train = pd.DataFrame(train, columns=['distanceKM', 'taxiDurationMin',
    #                                      'vehicleType', 'vehicleOption', 'weight'])
    #
    # khavar = train[train['vehicleType'] == 'khavar']
    # treili = train[train['vehicleType'] == 'treili']
    # joft = train[train['vehicleType'] == 'joft']
    # tak = train[train['vehicleType'] == 'tak']
    #
    # print(len(khavar), len(treili), len(joft), len(tak))

    # train_x = train.drop(['price'], axis=1)
    # train_y = train['price']
    # print(train_y)


load_files()
