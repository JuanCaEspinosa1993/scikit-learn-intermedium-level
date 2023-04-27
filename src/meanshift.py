import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == '__main__':
    df = pd.read_csv('dataset/candy.csv')

    X = df.drop('competitorname', axis=1)
    
    meanshift = MeanShift().fit(X)

    #Showing the number of used groups
    print(max(meanshift.labels_) + 1)
    print("="*60)
    #Showing the centers applyed for our algorithm
    print(meanshift.cluster_centers_)

    df['meanshit'] = meanshift.labels_
    print("="*60)
    print(df.head(20))