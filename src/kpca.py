import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from joblib import dump

if __name__ == '__main__':
    df_heart = pd.read_csv('dataset/heart.csv')
    
    #splitting the dataset in features a target variables
    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    #Aplying Scaling
    df_features = StandardScaler().fit_transform(df_features)

    #Splitting the dataset in train ant teset set
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)
   
   #Instancing KernelPCA. We are selecting the 4 components that provide the most information percentage 
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    df_train = kpca.transform(X_train)
    df_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(df_train, y_train)
    print('SCORE KPCA: ', logistic.score(df_test, y_test))

    #Saving the model
    dump(kpca, 'model/KPCA_model.pkl')


