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

with open("authenticom_sales.csv", newline='') as f:
    reader = csv.reader(f)
    X = list(reader)
