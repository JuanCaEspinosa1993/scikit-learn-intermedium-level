import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    df = pd. read_csv('dataset/heart.csv')

    X = df.drop(['target'], axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    #Here the n_estimator means How many trees we are going to use inside.
    boost = GradientBoostingClassifier(n_estimators=50)
    boost.fit(X_train, y_train)
    boost_pred = boost.predict(X_test)

    print("="*60)
    print(accuracy_score(y_test, boost_pred))
