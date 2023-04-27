import pandas as pd
#It similar tu Kmeans but less demanding  on computational resources
from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':
    df = pd.read_csv('dataset/candy.csv')

    X = df.drop('competitorname', axis=1)
    
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8)
    kmeans.fit(X)
    #Print the number of used clusters
    print("Total de centros: ", len(kmeans.cluster_centers_))
    print("="*60)
    print(kmeans.predict(X))
    
    #Adding the group for each record in our df creating a new column
    df['group'] = kmeans.predict(X)

    print(df.head(20))