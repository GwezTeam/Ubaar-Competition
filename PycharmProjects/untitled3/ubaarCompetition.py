import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_files():
    """load train and test data"""
    train = pd.read_csv("train.csv")
    x = train[['distanceKM', 'taxiDurationMin',
               'vehicleType', 'vehicleOption', 'weight']]
    x = np.reshape(x, (50000, 5))

    vehicleType = {'treili': 0, 'khavar': 1, 'joft': 2, 'tak': 3}
    vehicleOption = {'kafi': 0, 'mosaghaf_felezi': 1, 'kompressi': 2, 'bari': 3,
                     'labehdar': 4, 'yakhchali': 5, 'hichkodam': 6, 'mosaghaf_chadori': 7,
                     'transit_chadori': 8}

    x = encode(vehicleType, x, 'vehicleType')
    x = encode(vehicleOption, x, 'vehicleOption')

    # fill nan
    for column in x.columns:
        x[column] = x[column].fillna(np.mean(x[column]))

    # normalize data
    x = x.loc[:, ['distanceKM', 'taxiDurationMin',
                  'vehicleType', 'vehicleOption', 'weight']]
    # print(x)
    x = StandardScaler().fit_transform(x)
    x = pd.DataFrame(x, columns=['distanceKM', 'taxiDurationMin',
                                 'vehicleType', 'vehicleOption', 'weight'])

    # # PCA
    # new_column = twoD_PCA(x[['distanceKM', 'taxiDurationMin']])
    # x = x.drop(['distanceKM', 'taxiDurationMin'], axis=1)
    # x = pd.concat([x, twoD_PCA(x)], axis=1)

    # print(x)
    y = train.loc[:, train.columns == 'price']
    y = np.reshape(y, (50000, 1))

    train_x = x[0:34999]
    train_y = y[0:34999]
    test_x = x[35000:]
    test_y = y[35000:]

    return train_x, train_y, test_x, test_y


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
    data = pd.read_csv('test.csv')

    x = data[['date', 'sourceLatitude', 'sourceLongitude',
              'destinationLatitude', 'destinationLongitude',
              'distanceKM', 'taxiDurationMin',
              'vehicleType', 'vehicleOption', 'weight']]
    id = data[['ID']]
    vehicleType = {'treili': 0, 'khavar': 1, 'joft': 2, 'tak': 3}
    vehicleOption = {'kafi': 0, 'mosaghaf_felezi': 1, 'kompressi': 2, 'bari': 3,
                     'labehdar': 4, 'yakhchali': 5, 'hichkodam': 6, 'mosaghaf_chadori': 7,
                     'transit_chadori': 8}
    x = encode(vehicleType, x, 'vehicleType')
    x = encode(vehicleOption, x, 'vehicleOption')

    return x, id


def mean_absolute_percentage_error(true_y, pred_y):
    nptrue = np.reshape(true_y, (1, len(true_y)))
    nppred = np.reshape(pred_y, (1, len(pred_y)))
    mean = np.mean(np.abs((nptrue - nppred) / nptrue)) * 100
    return mean


train_x, train_y, test_x, test_y = load_files()

# sb.pairplot(train_x)
# plt.show()



train_x = train_x.values
train_y = train_y.values
test_x = test_x.values
test_y = test_y.values

# print(type(train_x))

regression = linear_model.LinearRegression()
regression.fit(train_x, train_y)

test, id_test = load_data()
# print(id_test)
id_list = []
for list in id_test.loc[:, 'ID']:
    id_list.append(list)

y_pred = regression.predict(test_x)
y_list = []
for list in y_pred:
    if list[0] < 0:
        list[0] = -list[0]
    y_list.append(int(list[0]))

# print(id_list)

# y_pred = regression.predict(test_x)
s = pd.Series(y_list)
# print(id_list)

print(mean_absolute_percentage_error(test_y, y_list))
# print(id_test.values)

# print(len(id_list), len(y_list))
df = pd.DataFrame({'ID': id_list, 'price': y_list})

# print(df.describe())

# writer = pd.ExcelWriter('submission.csv')
df.reset_index()
df.to_csv('submisiion.csv', index=False)
print(df.describe())
# print(df)
