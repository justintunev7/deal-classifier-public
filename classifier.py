from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv


def split_data(X, y, test_split=.15):
    # split into test/train data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split)
    return X_train, X_test, y_train, y_test

def dt_grid_search(X_train, X_test, y_train, y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    print("SCORE:", dtc.score(X_test, y_test))

def knn_grid_search(X_train, X_test, y_train, y_test):
    p=1
    k = [1,3,5,7,9]
    for _k in k:
        knn = KNeighborsClassifier(n_neighbors=_k, weights='uniform', algorithm='auto',
                                    p=p, metric='minkowski', metric_params=None, n_jobs=None)
        knn.fit(X_train, y_train)
        print("SCORE:", knn.score(X_test, y_test), "\tk:", _k, "\tP:", p)


def mlp_grid_search(X_train, X_test, y_train, y_test):
    lr = [.0001, .001, .1]
    momentum = .99
    alpha = .0001
    layers = [(50, 50), (20,20), (100), (10,10,20,10)]

    for _lr in lr:
        for layer in layers:
            mlp = MLPClassifier(hidden_layer_sizes=layer,
                                solver='sgd',
                                activation='relu',
                                learning_rate_init=_lr,
                                momentum=momentum,
                                max_iter=200,
                                nesterovs_momentum=False,
                                early_stopping=False,
                                alpha=alpha)
            mlp.fit(X_train, y_train)
            print("SCORE:", mlp.score(X_test, y_test),
                    "\tLayers:", layer, "\tLR:", _lr)
                
def ml_algorithm_search(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(10,10), learning_rate_init=.001)
    knn = KNeighborsClassifier(n_neighbors=5)
    dt = DecisionTreeClassifier()
    mlp.fit(X_train, y_train)
    print("SCORE Multi-Layered Perceptron:", mlp.score(X_test, y_test))
    knn.fit(X_train, y_train)
    print("SCORE K-Nearest Neighbors:", knn.score(X_test, y_test))
    dt.fit(X_train, y_train)
    print("SCORE Decision Tree:", dt.score(X_test, y_test))

def one_hot_encode_labels(labels):
  return preprocessing.LabelBinarizer().fit(labels).transform(labels)

def one_hot_encode(data, categorical_feat=[2, 11, 12, 15, 16, 17, 18, 19]):
    non_categorical = list(set(range(len(data[0]))) - set(categorical_feat))
    enc = OneHotEncoder()
    enc.fit(data[:, categorical_feat])
    encoded = np.array(enc.transform(data[:, categorical_feat]).toarray())
    return np.column_stack((data[:, non_categorical], encoded))

def preprocess(data):
    data = np.array(data[1:])
    data[data == 'NULL'] = "-1"

    # convert columns to binary where we only care about if the value exists
    binary_indexes = [16,17]
    data[:,binary_indexes] = np.array([data[:,binary_indexes] == np.full((len(data), len(binary_indexes)),"-1")])
    data = convert_date(data)
    X = normalize(one_hot_encode(data[:,:-1]).astype(float))
    y = one_hot_encode_labels(data[:,-1]).astype(float)
    # np.savetxt("data_dump_X.csv", X, delimiter=",")
    # np.savetxt("data_dump_y.csv", y, delimiter=",")
    return split_data(X, y)


def normalize(data):
    return np.array(Normalizer().transform(np.array(data, dtype=float)), dtype=float)

def convert_date(data, date_index=3):
    origin_date = datetime.datetime(1970, 1, 1)
    format_str = '%m/%d/%Y'  # The format
    for row in range(len(data)):
        date = data[row, date_index]
        if date != '-1':
            dt = datetime.datetime.strptime(date, format_str)
            data[row, date_index] = (dt - origin_date).total_seconds()
    return data

use_data_dump = False
if use_data_dump:
    with open("../data_dump_X.csv", newline='') as f:
        reader = csv.reader(f)
        X = list(reader)
    with open("../data_dump_y.csv", newline='') as f:
        reader = csv.reader(f)
        y = list(reader)
    X_train, X_test, y_train, y_test = split_data(np.array(X, dtype=float), np.array(y, dtype=float))
    X_train, X_test = normalize(X_train), normalize(X_test)
else:
    with open("authenticom_sales_extra_10000.csv", newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    categorical_features = [2, 11, 12, 15, 16, 17, 18, 19] # 20
    X_train, X_test, y_train, y_test = preprocess(data)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# mlp_grid_search(X_train, X_test, y_train, y_test)
# knn_grid_search(X_train, X_test, y_train, y_test)
# dt_grid_search(X_train, X_test, y_train, y_test)
ml_algorithm_search(X_train, X_test, y_train, y_test)
