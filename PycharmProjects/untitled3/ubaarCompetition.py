import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow
import xgboost as xgb


vehicleType = {'treili': 0, 'khavar': 1, 'joft': 2, 'tak': 3}
vehicleOption = {'kafi': 9, 'mosaghaf_felezi': 1, 'kompressi': 2, 'bari': 3,
                 'labehdar': 4, 'yakhchali': 5, 'hichkodam': 6, 'mosaghaf_chadori': 7,
                 'transit_chadori': 8}
state = {'تهران': 1310, 'اصفهان': 230, 'فارس': 29, 'همدان': 28,
               'البرز': 27, 'گیلان': 26, 'زنجان': 25, 'چهارمحال و بختیاری': 24,
               'کردستان': 23, 'کرمان': 22, 'یزد': 21, 'لرستان': 20,
               'آذربایجان شرقی': 19, 'خراسان رضوی': 18, 'کرمانشاه': 17,
               'قزوین': 16, 'مرکزی': 15, 'سمنان': 14, 'گلستان': 13,
               'سیستان و بلوچستان': 12, 'خوزستان': 11, 'بوشهر': 10,
               'ایلام': 9, 'اردبیل': 8, 'قم': 7, 'مازندران': 6,
               'هرمزگان': 5, 'آذربایجان غربی': 4, 'خراسان شمالی': 3,
               'کهگیلویه و بویراحمد': 2, 'خراسان جنوبی': 1}


def load_files():
    """load train and test data"""
    train = pd.read_csv("train.csv")
    x = train[['ID', 'sourceLatitude', 'sourceLongitude', 'destinationLatitude',
               'destinationLongitude', 'SourceState',
               'destinationState', 'distanceKM', 'taxiDurationMin',
               'vehicleType', 'vehicleOption', 'weight', 'price']]
    x = np.reshape(x, (50000, 13))

    x = encode(state, x, 'SourceState')
    x = encode(state, x, 'destinationState')
    x = encode(vehicleType, x, 'vehicleType')
    x = encode(vehicleOption, x, 'vehicleOption')

    # fill nan
    for column in x.columns:
        x[column] = x[column].fillna(np.mean(x[column]))

    # # normalize data
    # x = pd.concat([x[['ID']], x.loc[:, ['distanceKM', 'taxiDurationMin',
    #               'vehicleType', 'vehicleOption', 'weight', 'price']]], axis=1)
    #
    # # print(np.shape(x))
    # # print(x)
    # x = StandardScaler().fit_transform(x)
    # x = pd.DataFrame(x, columns=['ID', 'distanceKM', 'taxiDurationMin',
    #                              'vehicleType', 'vehicleOption', 'weight', 'price'])

    # # PCA
    # new_column = twoD_PCA(x[['distanceKM', 'taxiDurationMin']])
    # x = x.drop(['distanceKM', 'taxiDurationMin'], axis=1)
    # x = pd.concat([x, twoD_PCA(x)], axis=1)

    # print(x)
    # y = train.loc[:, train.columns == 'price']
    # y = np.reshape(y, (50000, 1))

    train_x = x[0:34999]
    test_x = x[35000:]

    return x, test_x


def encode(dic, dataFrame, column_name):
    dataFrame.loc[:, dataFrame.columns == column_name] = dataFrame[[column_name]] \
        .applymap(lambda z: dic[z])
    return dataFrame


def twoD_PCA(dataFrame):
    pca = PCA(n_components=1)
    p_components = pca.fit_transform(dataFrame)
    pdf = pd.DataFrame(data=p_components, columns=['time-distance'])
    return pdf


def load_data():
    test = pd.read_csv("test.csv")
    x = test[['ID', 'sourceLatitude', 'sourceLongitude', 'destinationLatitude',
               'destinationLongitude', 'SourceState',
               'destinationState', 'distanceKM', 'taxiDurationMin',
               'vehicleType', 'vehicleOption', 'weight']]
    x = np.reshape(x, (15000, 12))

    x = encode(vehicleType, x, 'vehicleType')
    x = encode(vehicleOption, x, 'vehicleOption')
    x = encode(state, x, 'SourceState')
    x = encode(state, x, 'destinationState')
    # fill nan
    for column in x.columns:
        x[column] = x[column].fillna(np.mean(x[column]))

    # # normalize data
    # x = x.loc[:, ['distanceKM', 'taxiDurationMin',
    #               'vehicleType', 'vehicleOption', 'weight']]
    # # print(x)
    # x = StandardScaler().fit_transform(x)
    # x = pd.DataFrame(x, columns=['distanceKM', 'taxiDurationMin',
    #                              'vehicleType', 'vehicleOption', 'weight'])
    return x


def mean_absolute_percentage_error(true_y, pred_y):
    nptrue = np.reshape(true_y, (1, len(true_y)))
    nppred = np.reshape(pred_y, (1, len(pred_y)))
    mean = np.mean(np.abs((nptrue - nppred) / nptrue)) * 100
    return mean


def classifier(dataFrame):
    khavar = dataFrame[dataFrame['vehicleType'] == 0]
    treili = dataFrame[dataFrame['vehicleType'] == 1]
    joft = dataFrame[dataFrame['vehicleType'] == 2]
    tak = dataFrame[dataFrame['vehicleType'] == 3]
    return khavar, treili, joft, tak


def seperate_x_from_y(dataFrame):
    return dataFrame.drop(labels=['ID', 'price'], axis=1), dataFrame['price'], dataFrame['ID']


def seperate_x_from_id(dataFrame):
    return dataFrame.drop(labels=['ID'], axis=1), dataFrame['ID']


def regression(x, y, tx, ty=None):
    regression = linear_model.LinearRegression()
    regression.fit(x, y)
    y_pred = regression.predict(tx)
    # print(y_pred)
    yp_list = y_pred.tolist()
    if ty is not None:
        ty_list = ty.tolist()
        print(mean_absolute_percentage_error(ty_list, yp_list))
        # print(y_pred)
    return y_pred


def concat_predic_and_ID(price, id):
    return pd.DataFrame({'ID': id, 'price': price})


def fill_zero(df):
    df['price'] = df['price'] / df['weight']
    df_non_zero = df[df['SourceState'] == df['destinationState']]
    df_non_zero = df_non_zero[df_non_zero['distanceKM'] != 0]
    price_mean = df_non_zero.groupby(['vehicleType']).price.mean()
    distance_mean = df_non_zero.groupby(['vehicleType']).distanceKM.mean()
    idx_zero_distnce = df[df['distanceKM'] == 0].index
    print(price_mean)

    print(distance_mean)
    for i in idx_zero_distnce:
        df.loc[i, 'distanceKM'] = (distance_mean[df.loc[i, 'vehicleType']] * df.loc[i, 'price']) /\
                                  price_mean[df.loc[i, 'vehicleType']]

    return df


def xgboost(x, y, tx, ty=None):
    params = {'eta': 0.3, 'max_depth': 10, 'objective': 'reg:linear',
              'eval_metric': 'mae', 'silent': False}

    watchlist = [(xgb.DMatrix(x, y), 'train')]

    xgb_model = xgb.train(params, xgb.DMatrix(x, y), 100, watchlist, verbose_eval=10, maximize=False, early_stopping_rounds=20)

    pred = xgb_model.predict(xgb.DMatrix(tx), ntree_limit=xgb_model.best_ntree_limit)
    pred = [x for x in pred]

    if ty is not None:
        true = ty.tolist()
        print(mean_absolute_percentage_error(true, pred))
    return pred


train, test = load_files()
khavar, treili, joft, tak = classifier(train)

khavar_x, khavar_y, khavar_ID = seperate_x_from_y(khavar)
treili_x, treili_y, treili_ID = seperate_x_from_y(treili)
joft_x, joft_y, joft_ID = seperate_x_from_y(joft)
tak_x, tak_y, tak_ID = seperate_x_from_y(tak)

# khavar_test, treili_test, joft_test, tak_test = classifier(test)
# khavar_tx, khavar_ty, khavar_tID = seperate_x_from_y(khavar_test)
# treili_tx, treili_ty, treili_tID = seperate_x_from_y(treili_test)
# joft_tx, joft_ty, joft_tID = seperate_x_from_y(joft_test)
# tak_tx, tak_ty, tak_tID = seperate_x_from_y(tak_test)

true_test = load_data()
khavar_test, treili_test, joft_test, tak_test = classifier(true_test)

khavar_tx, khavar_tID = seperate_x_from_id(khavar_test)
treili_tx, treili_tID = seperate_x_from_id(treili_test)
joft_tx, joft_tID = seperate_x_from_id(joft_test)
tak_tx, tak_tID = seperate_x_from_id(tak_test)


pred_khavar = xgboost(khavar_x, khavar_y, khavar_tx)
pred_khavar = [round(z) for z in pred_khavar]
print(pred_khavar)
pred_treili = xgboost(treili_x, treili_y, treili_tx)
pred_treili = [round(z) for z in pred_treili]
pred_joft = xgboost(joft_x, joft_y, joft_tx)
pred_joft = [round(z) for z in pred_joft]
pred_tak = xgboost(tak_x, tak_y, tak_tx)
pred_tak = [round(z) for z in pred_tak]



"""Neuarl Network"""
# # define and fit the final model
# input_size = 11
# #khavar
# model = Sequential()
# model.add(Dense(18, input_dim=input_size, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mean_absolute_percentage_error', optimizer='rmsprop')
# model.fit(khavar_x, khavar_y, epochs=50, verbose=1)
#
# pred_khavar = model.predict(khavar_tx)
# pred_khavar = pred_khavar.tolist()
# pred_khavar = [int(z[0]) for z in pred_khavar]
# # print(pred_khavar)
#
# ty = khavar_ty.tolist()
# print("khavar : " + str(mean_absolute_percentage_error(ty, pred_khavar)))
# # exit()
# #joft
# model = Sequential()
# model.add(Dense(64, input_dim=input_size, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mean_absolute_percentage_error', optimizer='rmsprop')
# model.fit(joft_x, joft_y, epochs=50, verbose=1)
#
# pred_joft = model.predict(joft_tx)
# pred_joft = pred_joft.tolist()
# pred_joft = [int(z[0]) for z in pred_joft]
# ty = joft_ty.tolist()
# print("joft : " + str(mean_absolute_percentage_error(ty, pred_joft)))
# # exit()
#
# # #treili
# model = Sequential()
# model.add(Dense(18, input_dim=input_size, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mean_absolute_percentage_error', optimizer='rmsprop')
# model.fit(treili_x, treili_y, epochs=50, verbose=1)
#
# pred_treili = model.predict(treili_tx)
# pred_treili = pred_treili.tolist()
# pred_treili = [int(z[0]) for z in pred_treili]
# ty = treili_ty.tolist()
# print("treili : " + str(mean_absolute_percentage_error(ty, pred_treili)))
#
#
# #tak
# model = Sequential()
# model.add(Dense(18, input_dim=input_size, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mean_absolute_percentage_error', optimizer='rmsprop')
# model.fit(tak_x, tak_y, epochs=50, verbose=1)
#
# pred_tak = model.predict(tak_tx)
# pred_tak = pred_tak.tolist()
# pred_tak = [int(z[0]) for z in pred_tak]
# ty = tak_ty.tolist()
# print("tak : " + str(mean_absolute_percentage_error(ty, pred_tak)))


"""linear regression"""
# pred_khavar = regression(khavar_x, khavar_y, khavar_tx)
# print(pred_khavar)
# pred_khavar = [int(x) for x in pred_khavar]
# pred_treili = regression(treili_x, treili_y, treili_tx)
# pred_treili = [int(x) for x in pred_treili]
# pred_joft = regression(joft_x, joft_y, joft_tx)
# pred_joft = [int(x) for x in pred_joft]
# pred_tak = regression(tak_x, tak_y, tak_tx)
# pred_tak = [int(x) for x in pred_tak]
# final_khavar = pd.DataFrame({'price': pred_khavar})
# final_treili = pd.DataFrame({'price': pred_treili})
# final_joft = pd.DataFrame({'price': pred_joft})
# final_tak = pd.DataFrame({'price': pred_tak})
# final_y_df = pd.concat([final_khavar, final_treili, final_joft, final_tak])
# # final_y_df = final_y_df[:, 1]
# final_list = final_y_df['price'].tolist()
# final_list = [int(x[0]) for x in final_list]


# final_list = pred_khavar + pred_treili + pred_joft + pred_tak
# test_y = khavar_ty.append(treili_ty)
# test_y = test_y.append(joft_ty)
# test_y = test_y.append(tak_ty)
# print(test_y, final_list)
# print(np.mean(np.abs((test_y - final_list) / test_y)) * 100)


df_khavar = concat_predic_and_ID(pred_khavar, khavar_tID)
df_treili = concat_predic_and_ID(pred_treili, treili_tID)
df = pd.concat([df_khavar, df_treili])
df = pd.concat([df, concat_predic_and_ID(pred_joft, joft_tID)])
df = pd.concat([df, concat_predic_and_ID(pred_tak, tak_tID)])

df.to_csv('submisiion.csv', index=False)


print(df.describe())
