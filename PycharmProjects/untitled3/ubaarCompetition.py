import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier


def load_files():
    """load train and test data"""
    train = pd.read_csv("train.csv")
    x = train[['ID','distanceKM', 'taxiDurationMin',
               'vehicleType', 'vehicleOption', 'weight', 'price']]
    x = np.reshape(x, (50000, 7))

    vehicleType = {'treili': 0, 'khavar': 1, 'joft': 2, 'tak': 3}
    vehicleOption = {'kafi': 9, 'mosaghaf_felezi': 1, 'kompressi': 2, 'bari': 3,
                     'labehdar': 4, 'yakhchali': 5, 'hichkodam': 6, 'mosaghaf_chadori': 7,
                     'transit_chadori': 8}

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

    return train_x, test_x


def encode(dic, df, column_name):
    df.loc[:, df.columns == column_name] = df[[column_name]] \
        .applymap(lambda z: dic[z])
    return df


def twoD_PCA(df):
    pca = PCA(n_components=1)
    p_components = pca.fit_transform(df)
    pdf = pd.DataFrame(data=p_components, columns=['time-distance'])
    return pdf


def load_data():
    test = pd.read_csv("test.csv")
    x = test[['ID', 'sourceLatitude', 'sourceLongitude', 'destinationLatitude', 'destinationLongitude','distanceKM', 'taxiDurationMin',
               'vehicleType', 'vehicleOption', 'weight']]
    x = np.reshape(x, (15000, 10))

    vehicleType = {'treili': 0, 'khavar': 1, 'joft': 2, 'tak': 3}
    vehicleOption = {'kafi': 0, 'mosaghaf_felezi': 1, 'kompressi': 2, 'bari': 3,
                     'labehdar': 4, 'yakhchali': 5, 'hichkodam': 6, 'mosaghaf_chadori': 7,
                     'transit_chadori': 8}

    x = encode(vehicleType, x, 'vehicleType')
    x = encode(vehicleOption, x, 'vehicleOption')

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


def classifier(df):
    khavar = df[df['vehicleType'] == 0]
    treili = df[df['vehicleType'] == 1]
    joft = df[df['vehicleType'] == 2]
    tak = df[df['vehicleType'] == 3]
    return khavar, treili, joft, tak


def seperate_x_from_y(df):
    return df.drop(labels=['ID', 'price'], axis=1), df['price'], df['ID']


def seperate_x_from_id(df):
    return df.drop(labels=['ID'], axis=1), df['ID']


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


train, test = load_files()

khavar, treili, joft, tak = classifier(train)

khavar_x, khavar_y, khavar_ID = seperate_x_from_y(khavar)
treili_x, treili_y, treili_ID = seperate_x_from_y(treili)
joft_x, joft_y, joft_ID = seperate_x_from_y(joft)
tak_x, tak_y, tak_ID = seperate_x_from_y(tak)

khavar_test, treili_test, joft_test, tak_test = classifier(test)
khavar_tx, khavar_ty, khavar_tID = seperate_x_from_y(khavar_test)
treili_tx, treili_ty, treili_tID = seperate_x_from_y(treili_test)
joft_tx, joft_ty, joft_tID = seperate_x_from_y(joft_test)
tak_tx, tak_ty, tak_tID = seperate_x_from_y(tak_test)


# true_test = load_data()
# khavar_test, treili_test, joft_test, tak_test = classifier(true_test)
#
# khavar_tx, khavar_tID = seperate_x_from_id(khavar_test)
# treili_tx, treili_tID = seperate_x_from_id(treili_test)
# joft_tx, joft_tID = seperate_x_from_id(joft_test)
# tak_tx, tak_tID = seperate_x_from_id(tak_test)

# pred_khavar = regression(khavar_x, khavar_y, khavar_tx)
# pred_khavar = [int(x) for x in pred_khavar]
# pred_treili = regression(treili_x, treili_y, treili_tx)
# pred_treili = [int(x) for x in pred_treili]
# pred_joft = regression(joft_x, joft_y, joft_tx)
# pred_joft = [int(x) for x in pred_joft]
# pred_tak = regression(tak_x, tak_y, tak_tx)
# pred_tak = [int(x) for x in pred_tak]

model = XGBClassifier()
train_x, train_y, train_id = seperate_x_from_y(train)
model.fit(train_x, train_y)
y_pred = model.predict(test)
prediction = [round(value) for value in y_pred]
accuracy = mean_absolute_percentage_error(train_y, prediction)
print(accuracy)


# final_khavar = pd.DataFrame({'price': pred_khavar})
# final_treili = pd.DataFrame({'price': pred_treili})
# final_joft = pd.DataFrame({'price': pred_joft})
# final_tak = pd.DataFrame({'price': pred_tak})
# final_y_df = pd.concat([final_khavar, final_treili, final_joft, final_tak])
# final_list = final_y_df['price'].tolist()
#
# #
# test_y = khavar_ty.append(treili_ty)
# test_y = test_y.append(joft_ty)
# test_y = test_y.append(tak_ty)
#
#
# # print(len(test_y), len(final_list))
# #
# print(np.mean(np.abs((test_y - final_list) / test_y)) * 100)
#
#
# df_khavar = concat_predic_and_ID(pred_khavar, khavar_tID)
# df_treili = concat_predic_and_ID(pred_treili, treili_tID)
# df = pd.concat([df_khavar, df_treili])
# df = pd.concat([df, concat_predic_and_ID(pred_joft, joft_tID)])
# df = pd.concat([df, concat_predic_and_ID(pred_tak, tak_tID)])
#
# df.to_csv('submisiion.csv', index=False)
#

# print(df.describe())


# sb.pairplot(train_x)
# plt.show()



# train_x = train_x.values
# train_y = train_y.values
# test_x = test_x.values
# test_y = test_y.values

# print(type(train_x))
#
# regression = linear_model.LinearRegression()
# regression.fit(train_x, train_y)
#
# test, id_test = load_data()
# # print(id_test)
# id_list = []
# for list in id_test.loc[:, 'ID']:
#     id_list.append(list)
#
#
# y_pred = regression.predict(test_x)
#
# y_list = []
# for list, i in zip(y_pred, range(len(y_pred))):
#     if list[0] < 0:
#         # list[0] = abs(list[0])
#         print(test.loc[i,:], test_y[i], list[0])
#     y_list.append(int(list[0]))
#
# # print(id_list)
#
# # y_pred = regression.predict(test_x)
# s = pd.Series(y_list)
# # print(id_list)
#
# # print(mean_absolute_percentage_error(test_y, y_list))
# # print(id_test.values)
#
# # print(len(id_list), len(y_list))
# df = pd.DataFrame({'ID': id_list, 'price': y_list})
#
# # print(df.describe())
#
# # writer = pd.ExcelWriter('submission.csv')
# df.reset_index()
# df.to_csv('submisiion.csv', index=False)
# print(df.describe())
# # print(df)
