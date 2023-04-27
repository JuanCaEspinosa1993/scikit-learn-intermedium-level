import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

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

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print("="*30)
    print(accuracy_score(y_test, knn_pred))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50)
    bag_class.fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print('='*30)
    print(accuracy_score(y_test, bag_pred))
    

    estimators = {
        'LogisticRegression' : LogisticRegression(),
        'SVC' : SVC(),
        'LinearSVC' : LinearSVC(),
        'DecisionTreeClf' : DecisionTreeClassifier(),
        'RandomForestClf' : RandomForestClassifier(random_state=0)
    }
    print("\n Here start bagging estimators testing\n")
    for name, estimator in estimators.items():
        model = BaggingClassifier(base_estimator=estimator, n_estimators=50)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("="*40)
        print(f"SCORE Bagging with {name}: {round(accuracy_score(y_test, predictions), 2)} ")
    
    print('Here We apply Cross-validation')

    CV =10
    cv_df = pd.DataFrame(index=range(CV*len(estimators)))
    entries =[]
    for name, model in estimators.items():
        accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)

        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((name, fold_idx, accuracy))
    
    cv_df = pd.DataFrame(entries, columns=['model_name','fold_idx', 'accuracy'])
    mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
    std_accuracy = cv_df.groupby('model_name').accuracy.std()
    acc = pd.concat([mean_accuracy,std_accuracy], axis=1, ignore_index=True)
    acc.columns = ['Mean Accuracy', 'Standard deviaiton']
    
    print('\n This is an accuracy more realistic. Thisi is the accuracy mean in Cross Validation \n')
    print(acc)