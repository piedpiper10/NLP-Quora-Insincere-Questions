import numpy
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
scaler = MinMaxScaler()


data=pd.read_csv('train_75.csv', delimiter=',', usecols = ['question_text','target'])
class_weights = class_weight.compute_class_weight('balanced',numpy.unique(data['target']),data['target'])
print class_weights

