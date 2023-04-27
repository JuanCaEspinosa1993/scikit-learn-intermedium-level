import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    df = pd.read_csv('dataset/felicidad.csv')

    X = df[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = df[['score']]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model_linear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = model_linear.predict(X_test)

    #if alpha is larger, the penalty is greater
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    #Showing the loss for each method. If the loss is less, the model is better
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss: ",linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss: ",lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss: ", ridge_loss)

    print("="*30)
    print("Coef LASSO")
    print(modelLasso.coef_)

    print("="*30)
    print("Coef RIDGE")
    print(modelRidge.coef_)
