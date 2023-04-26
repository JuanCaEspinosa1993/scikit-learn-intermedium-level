import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

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

    #Instancing PCA
    pca = PCA(n_components=3)
    pca.fit(X_train)

    #Instancing IncrementalPCA
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    #Creating our Logistic Regression model
    logistic = LogisticRegression(solver='lbfgs')

    #Applying PCA
    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)
    logistic.fit(df_train, y_train)
    print("SCORE PCA: ", logistic.score(df_test, y_test))

    #Applying IncrementalPCA    
    df_train = ipca.transform(X_train)
    df_test = ipca.transform(X_test)
    logistic.fit(df_train, y_train)
    print("SCORE IPCA: ", logistic.score(df_test, y_test))

    #Saving the model
    dump(pca, 'model/PCA_model.pkl')
    dump(pca, 'model/IncrementalPCA_model.pkl')

    #plotting the results
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.title('Comportamiento al reducir dimensiones')
    plt.xlabel('Componentes')
    plt.ylabel("Porcentaje de aportación")
    plt.savefig('images/PCA_dimensionality_reduction.png')