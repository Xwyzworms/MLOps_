#%%
import pandas as pd
import numpy as np
import datetime
import matplotlib
import sklearn

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics

#%%

np.random.seed(0)
def simulateRideDistances():
    rideDistances = np.concatenate([
        10 * np.random.random(size=470),
        30 * np.random.random(size=10),
        10 * np.random.random(size=10),
        10 * np.random.random(size=10)]
    )

    return rideDistances

def simulateSpeedDistances():
    speedDistances = np.concatenate([
        np.random.normal(loc=30, scale=5, size=470),
        np.random.normal(loc=30, scale=5, size=10),
        np.random.normal(loc=50, scale=10, size=10),
        np.random.normal(loc=15, scale=4, size=10)
    ])

    return speedDistances

def simulateRideData():
    rideDistances = simulateRideDistances()
    speedDistances = simulateSpeedDistances()
    timeDistances = rideDistances / speedDistances

    df = pd.DataFrame({
        'ride_dists' : rideDistances,
        'speed_dists' : speedDistances,
        'time_dists' : timeDistances
    })  

    ride_ids = datetime.datetime.now().strftime("%Y%m%d") + df.index.astype(str)
    df['ride_id'] =ride_ids
    return df

def clusterAndLabel(data, create_and_show_plot = True):
    data = StandardScaler().fit_transform(data)
    dbscan = DBSCAN(eps=0.3, min_samples=10).fit(data)

    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    
    core_samples_mask[dbscan.core_sample_indices_] = True
    labels = dbscan.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) ## -1 represent the outliesr
    n_noise = list(labels).count(-1) ## COunt the outliers
    runMetadata = {
        'nClusters' : n_clusters,
        'nNoise' : n_noise,
        'silhouetteCoefficient' : metrics.silhouette_score(data,labels), # Calculate how well the data points assigned to a cluster compare with another cluster
        'labels' : labels,
        'data' : data
    }

    return runMetadata


if __name__ == "__main__":
    df = simulateRideData()
    result = clusterAndLabel(df)
    x_df,labels = result['data'],result['labels']
    
    marker = np.array(['o' if i >= 0 else 'x' for i in labels])
    for xp,yp, m, in zip(x_df[:,0], x_df[:,1],marker) :
        plt.scatter(xp,yp,marker=m)
    plt.show()
