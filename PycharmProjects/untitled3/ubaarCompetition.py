import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def load_files():
    """load train and test data"""
    train = pd.read_csv("train.csv")
    train_x = train[['date', 'sourceLatitude', 'sourceLongitude',
                     'destinationLatitude', 'destinationLongitude',
                     'distanceKM', 'taxiDurationMin',
                     'vehicleType', 'vehicleOption', 'weight']]

    vehicleType = {'treili': 0, 'khavar': 1, 'joft': 2, 'tak': 3}
    vehicleOption = {'kafi': 0, 'mosaghaf_felezi': 1, 'kompressi': 2, 'bari': 3, 'labehdar': 4,
                     'yakhchali': 5, 'hichkodam': 6, 'mosaghaf_chadori': 7, 'transit_chadori': 8}
    train_x.loc[:, train_x.columns == 'vehicleType'] = train_x[['vehicleType']] \
        .applymap(lambda x: vehicleType[x])
    train_x.loc[:, train_x.columns == 'vehicleOption'] = train_x[['vehicleOption']] \
        .applymap(lambda x: vehicleOption[x])
    # print(train_x['vehicleOption'])
    train_y = train.loc[:, train.columns == 'price']
    train_x = np.reshape(train_x, (50000, 10))
    train_y = np.reshape(train_y, (50000, 1))
    # print(train_y)
    test = pd.read_csv("test.csv")
    test_x = test[['date', 'sourceLatitude', 'sourceLongitude',
                     'destinationLatitude', 'destinationLongitude',
                     'distanceKM', 'taxiDurationMin',
                     'vehicleType', 'vehicleOption', 'weight']]
    test_x.loc[:, test_x.columns == 'vehicleType'] = test_x[['vehicleType']] \
        .applymap(lambda x: vehicleType[x])
    test_x.loc[:, test_x.columns == 'vehicleOption'] = test_x[['vehicleOption']] \
        .applymap(lambda x: vehicleOption[x])

    test_y = test.loc[:, test.columns == 'price']


    train_x = np.reshape(train_x, (50000, 10))
    train_y = np.reshape(train_y, (50000, 1))
    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = load_files()

print(test_y)
#
train_x = train_x.values
test_x = test_x.values
for i in range(len(train_x)):
    for j in range(len(train_x[0])):
        if train_x[i][j] != train_x[i][j]:
            train_x[i][j] = 0


for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if test_x[i][j] != test_x[i][j]:
            test_x[i][j] = 0

regression = linear_model.LinearRegression()
regression.fit(train_x, train_y)

y_pred = regression.predict(test_x)
y_list = []
for list in y_pred:
    y_list.append(list[0])
# y_pred = np.reshape(y_pred, (15000,))
# test_y = np.reshape(test_y, (15000, 1))
# print(y_list)
# print(test_y)
# accuracy = mean_squared_error(y_list, test_y)
# print(accuracy)

