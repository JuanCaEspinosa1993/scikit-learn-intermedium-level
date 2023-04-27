import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold

if __name__ == '__main__':
    df = pd.read_csv("dataset/felicidad.csv")

    X = df.drop(["country", "score"], axis=1)
    y = df["score"]

    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error")
    print(score)
    print("="*60)
    #The  absoulte value smallest means the model is better
    print(f"El score mas realista es:  {np.mean(score)}")

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(df):
        print(train)
        print(test)

