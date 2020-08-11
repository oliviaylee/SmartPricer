import sys
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# DATASET PREPARATION: Training and Test Data Splitting
# Dataset Source: https://nijianmo.github.io/amazon/index.html
df = pd.read_csv('train.tsv', sep = '\t')
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
# To check, view first 5 rows with train.head(), and table info with train.info()


# DATA VISUALIZATION & ANALYSIS
# Viewing Price Data
plt.subplot(1, 2, 1)
(train['price']).plot.hist(bins=50, figsize=(12, 6), edgecolor = 'white', range = [0, 250])
plt.xlabel('price', fontsize=12)
plt.title('Price Distribution', fontsize=12)
# Log Transformation
plt.subplot(1, 2, 2)
np.log(train['price']+1).plot.hist(bins=50, figsize=(12,6), edgecolor='white')
plt.xlabel('log(price+1)', fontsize=12)
plt.title('Price Distribution', fontsize=12)


# MODEL TRAINING
# Other features to consider: feature salesRank, related, similar, image (for image recognition)
# Constants
NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000
# Data Cleaning
def handle_missing_inplace(dataset): 
    dataset['categories'].fillna(value='missing', inplace=True) 
    dataset['brand'].fillna(value='missing', inplace=True)
    dataset['description'].replace('No description yet,''missing', inplace=True) 
    dataset['description'].fillna(value='missing', inplace=True)

def cutting(dataset):
    pop_brand = dataset['brand'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand'].isin(pop_brand), 'brand'] = 'missing'
    pop_category = dataset['categories'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]

def to_categorical(dataset):
    dataset['categories'] = dataset['categories'].astype('category')
    dataset['brand'] = dataset['brand'].astype('category')
    dataset['condition'] = dataset['condition'].astype('category')

def main():
    test_new = test.drop('price', axis=1)
    y_test = np.log1p(test["price"])
    train = train[train.price != 0].reset_index(drop=True)
    nrow_train = train.shape[0]
    y = np.log1p(train["price"])

    merge: pd.DataFrame = pd.concat([train, test_new])
    handle_missing_inplace(merge)
    cutting(merge)
    to_categorical(merge)
    # To check, view first 5 rows with merge.head(), and table info with merge.info()

    # tf-idf vectorization of name and category names
    cv = CountVectorizer(min_df=NAME_MIN_DF)
    cv = CountVectorizer()
    X_category = cv.fit_transform(merge['categories'])
    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION, ngram_range=(1, 3), stop_words='english')
    X_description = tv.fit_transform(merge['description'])
    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand'])
    X_dummies = csr_matrix(pd.get_dummies(merge['condition'], sparse=True).values)

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X_name = cv.fit_transform(merge['name'])

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    train_X = lgb.Dataset(X, label=y)
    params = {
            'learning_rate': 0.66,
            'application': 'regression',
            'max_depth': 3,
            'num_leaves': 100,
            'verbosity': -1,
            'metric': 'RMSE',
        }
    gbm = lgb.train(params, train_set=train_X, num_boost_round=3200, verbose_eval=100)

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print(y_pred)
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(y_test, y_pred) ** 0.5))

if __name__ == "__main__":
    main()
