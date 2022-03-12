""""
Bryan Yekelchik
Lehigh CSE 447
Project 1
biy320@lehigh.edu
---------
##### To mount the driver of the data, please see main() #####
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def distance(a,b):
    """
    Given two np.arrays a and b
    ---------
    Return euclidean distance between a and b
    """
    return np.sqrt(sum(np.square(a-b)))

def initiate_centroids(k,data,init_random):
    """
    Given k clusters and the data set
    ---------
    Return the initial k centroids
    """
    centroids = []
    for i in range(k):
        if init_random == True:
            centroids.append(data[np.random.randint(0,len(data))]) #Randomly pick centroid index
        else:
            centroids.append(data[i])
    return np.asarray(centroids)

def cluster_assignment(data,centroids):
    """
    Given Centroids
    ---------
    Returns assign data to k clusters
    """
    cluster = np.zeros(len(data))
    for point in range(len(data)):
        distances =[distance(data[point],centroid) for centroid in centroids] # computing distance from each point in dataset to each centroid
        index = np.argmin(distances) #assigning each point to closest centroid
        cluster[point] = index
    return cluster

def compute_centroids(k, data, cluster):
    """
    Given data, number of cluster, and the current clusters
    ---------
    Return new computed centroids
    """
    new_centroids = []
    for i in range(k):
        node = []
        for j in range(len(data)):
            if cluster[j]==i:
                node.append(data[j])
        new_centroids.append(np.mean(node, axis=0))
    return np.array(new_centroids,dtype=object)

def measure_change(centroids_current,new_centroids):
    """
    Calculate the distance between previous centroids and current
    """
    delta = 0
    for a,b in zip(centroids_current,new_centroids):
        delta+=distance(a,b)
    return delta

def kmeans_clustering(data,tol,k,init_random,match_truth_values=False, truth_values= []):
    np.random.seed(10)
    """
    Kmeans clustering algorithm
    Input is numpy array, tolerence, max_iterations, k # of cluster, and centroid init_type 
    ---------
    Optional Arguments:
    
    match_truth Values: Boolean
    truth_values: values to match 
    ---------
    Returns list of assigned cluster for each element
    """
    centroids_current = initiate_centroids(k,data,init_random)
    change = 50
    while change > tol: #stopping condition
        cluster = cluster_assignment(data,centroids_current) #assign clusters using centroids_current
        new_centroids = compute_centroids(k, data, cluster) # calcualte new centroids with the latest assignemnts
        change = measure_change(centroids_current,new_centroids) # calculate the distance between previous centroids using eulcidean distance function
        centroids_current = new_centroids #assign new centroids as the 'current' ones and check stopping condition at the top
    
    if match_truth_values == True:
        cluster_matched = np.empty_like(cluster)
            # For each cluster label...
        for k in np.unique(cluster):

            # ...find and assign the best-matching truth label
            match_nums = [np.sum((cluster==k)*(truth_values==t)) for t in np.unique(truth_values)]
            cluster_matched[cluster==k] = np.unique(truth_values)[np.argmax(match_nums)]
        cluster = cluster_matched
    else:
        pass

    return cluster, centroids_current

# #### Fit Kmeans to Both Datasets and Graph

def plot_2d_clusters(data,clusters,centroids,type):
    """"
    Given pandas df, list of clusters, list of centroids, and type of clustering
    ---------
    Returns plot of clusters
    """    
    data[f'{type}_cluster'] = clusters
    clustered_data = data.groupby(f'{type}_cluster')
    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)
    for name, group in clustered_data:
        ax.plot(group.x, group.y, marker='o', linestyle='', label=name)
    for center in range(len(centroids)):
        ax.plot(centroids[center][:2][0],centroids[center][:2][1],marker = 'p',color = 'black',markersize = 15)
    ax.legend(numpoints=1, loc='upper left')
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    ax.set_title(f'Scatter Plot of data w/ \n Color Coded by Cluster {type}')
    return fig.show()

def similarity_calc(data,type,sigma,epsilon):
    type = type.lower()
    """
    Given two np array, the type of similarity calculation, and threshold similarity epsilon
    ---------
    Return the appropriate Similarity Matrix
    """
    sim_matrix = np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            if type == 'cosine':
                sim_matrix[i,j] = np.dot(data[i],data[j])/(np.linalg.norm(data[i])*(np.linalg.norm(data[j])))
            elif type == 'gaussian':
                sim_matrix[i,j] = np.exp((-distance(data[i],data[j])**2)/(2*sigma**2))
            else:
                print("please choose 'gaussian' or 'cosine'")
                break
    return np.where(sim_matrix > epsilon,sim_matrix,0)

def degree_matrix(sim_matrix):
    """"
    Given a similarity matrix
    --------
    Returns the degree matrix
    """
    d_matrix = np.zeros((len(sim_matrix),len(sim_matrix)))
    for i in range(len(sim_matrix)):
        d_matrix[i,i] = sum(sim_matrix[i])    
    return d_matrix

def spectral_clustering(data,k,tol,normalization,similarity,init_random,sigma,epsilon, truth_values=False, match_truth_values=[]):
    """
    Spectral clustering algorithm
    Input is numpy array, tolerence,type of normalization and similarity, and k # of cluster
    ---------
    Cluster initiation is random

    Optional Arguments:
    
    match_truth Values: Boolean
    truth_values: values to match 
    ---------
    Returns list of assigned cluster for each element
    """
    s_matrix = similarity_calc(data,similarity,sigma,epsilon)
    d_matrix = degree_matrix(s_matrix)
    l = d_matrix - s_matrix
    if normalization == True:
        # d_matrix_frac = fractional_matrix_power(d_matrix, -0.5)
        # l = (d_matrix_frac) * l * (d_matrix_frac)
        l = np.linalg.inv(d_matrix) *l
    else: 
        pass
    eig_val,eig_vec = np.linalg.eig(l)
    eig_val_non_zero = eig_val[np.where(eig_val>0)]
    k_small_eigval_nonzero = np.sort(eig_val_non_zero)[:k] #sort the list and get the first k elements
    k_small_eigval_nonzero_idx =[]
    for i in range(len(k_small_eigval_nonzero)):
        k_small_eigval_nonzero_idx.append(np.where((eig_val == k_small_eigval_nonzero[i]))) #find the idx of where these elements are in the eig_val matrix
    data = eig_vec[:,k_small_eigval_nonzero_idx] #pull out the the two idx corresponding to smallest non-zero eig_val
    data= data.reshape(data.shape[0],data.shape[1])
    cluster = kmeans_clustering(data,tol,k,init_random=True)
    return cluster
# #### Fit Spectral to Both Datasets and Graph

def generate_spectral_cluster(data,k,random_init,sigma,similarity='gaussian',epsilon =.3,normalization = False):
    """
    Given generated data, k clusters, random initalization of centroids,epsilon,normalization,similarity, sigma
    ---------
    Returns a graph of the fitted clusters and centroids 
    """
    sc = StandardScaler()
    if data =='square':
        # square
        data_input = pd.read_csv(r"C:\Users\Byeke\OneDrive\Documents\Spring_2022_Lehigh\data_mining\project_1\Datasets\square.txt",delimiter=' ',names = ['x','y']) #mount the driver here
        data_clustering = data_input.values
    elif data == 'elliptical':
        #elliptical
        data_input = pd.read_csv(r"C:\Users\Byeke\OneDrive\Documents\Spring_2022_Lehigh\data_mining\project_1\Datasets\elliptical.txt",delimiter=' ',names = ['x','y']) #mount the driver here
        data_clustering = sc.fit_transform(data_input)
    result_spectral, result_spectral_centroids = spectral_clustering(data_clustering,sigma = sigma,tol=.001,k=k,normalization = normalization,similarity= similarity,init_random = random_init,epsilon=epsilon)
    return result_spectral, result_spectral_centroids

# ### Real-World Dataset

def real_world_kmeans(data,data_name,remove_outliers=False):
    real_set_cols = data.columns.tolist()
    real_set_cols[0] = "gene_id"
    real_set_cols[1] = "ground_truth"
    data.columns = real_set_cols
    data.set_index('gene_id',inplace=True)
    if (remove_outliers == True and data_name.lower() == "iyer"):
        data.ground_truth.unique()
        data_no_outliers = data[data['ground_truth']!=-1]
        data = data_no_outliers
    data_clustering = data.iloc[:,1:]
    actual_k_data = data.ground_truth.unique().shape[0]
    sc = StandardScaler()
    scaled_values = sc.fit_transform(data_clustering)
    kmeans_clusters,kmeans_centroids = kmeans_clustering(scaled_values, .001,actual_k_data, init_random = True, match_truth_values=True, truth_values=data.ground_truth)
    return kmeans_clusters,kmeans_centroids,data

def real_world_spectral(data,data_name,remove_outliers, tol=.001, normalization=False, similarity='cosine', init_random=True, sigma = 2,epsilon =.5, match_truth_values=True):
    real_set_cols = data.columns.tolist()
    real_set_cols[0] = "gene_id"
    real_set_cols[1] = "ground_truth"
    data.columns = real_set_cols
    data.set_index('gene_id',inplace=True)
    print(data)
    if (remove_outliers == True and data_name.lower() == "iyer"):
        data.ground_truth.unique()
        data_no_outliers = data[data['ground_truth']!=-1]
        data = data_no_outliers
    data_clustering = data.iloc[:,1:]
    actual_k_data = data.ground_truth.unique().shape[0]
    sc = StandardScaler()
    scaled_values = sc.fit_transform(data_clustering)
    spectral_clusters,centroids_spectral = spectral_clustering(scaled_values, k=actual_k_data, tol=tol, normalization=normalization, similarity=similarity, init_random=init_random,
                                                            sigma = sigma,epsilon =epsilon, match_truth_values=match_truth_values, truth_values=data.ground_truth)
    return spectral_clusters,centroids_spectral,data

"""Number of true clusters for each dataset"""
# actual_k_cho = cho.ground_truth.unique().shape[0]
# actual_k_iyer = iyer.ground_truth.unique().shape[0]
# print(f'# Clusters Cho: {actual_k_cho} \n# Clusters iyer: {actual_k_iyer}')

# kmeans_iyer,kmeans_iyer_centroids = kmeans_clustering(iyer_no_outliers.iloc[:,1:].values, .001,actual_k_iyer, init_random = True, match_truth_values=True, truth_values=iyer_no_outlier_truth)
# kmeans_cho,kmeans_cho_centroids = kmeans_clustering(cho.iloc[:,:1].values, .001, actual_k_cho, init_random= True, match_truth_values=True, truth_values=cho_truth)

# spectral_iyer,centroids_iyer_spectral = spectral_clustering(iyer_sc_values, k=actual_k_iyer, tol=.001, normalization=False, similarity='cosine', init_random=True,
#                                                             sigma = 2,epsilon =.5, match_truth_values=True, truth_values=iyer_no_outlier_truth)

# spectral_cho,centroids_cho_spectral = spectral_clustering(cho_sc_values, k=actual_k_cho, tol=.001, normalization=False, similarity='gaussian', init_random=True,
#                                                             sigma = 2,epsilon =.5,match_truth_values=True, truth_values=cho.ground_truth)

def sse_calc(data):
    """
    Given data with labeled clusters 
    --------
    Returns the sse
    """
    dist = 0
    n_clusters = data.spectral_label.unique().tolist()

    for i in range(len(n_clusters)): #for each cluster
        mean_i = data[data['spectral_label']==n_clusters[i]].mean().iloc[1:-1].to_numpy() #get the ith centroid
        cluster_data = data[data['spectral_label']==n_clusters[i]] #get the data labeled into the ith cluster

        for point in range(len(cluster_data)): #for each point in the cluster
            b = cluster_data.iloc[point,1:-1].to_numpy() #get each point
            dist += distance(mean_i,b) #calculate euclidean distance between centroid i and each point in the ith cluster  
    return dist

# result_spectral, result_spectral_centroids = generate_spectral_cluster('square',k=2,random_init=True,sigma=1,similarity='gaussian',epsilon =.5,normalization = False)
# plot_2d_clusters(square, result_spectral, result_spectral_centroids,"spectral")
# result_spectral, result_spectral_centroids = generate_spectral_cluster('elliptical',k=2,random_init=True,sigma=1,similarity='gaussian',epsilon =.75,normalization = False)
# plot_2d_clusters(elliptical, result_spectral, result_spectral_centroids,"spectral")

def main(dataset,cluster_type,remove_outliers):
    if dataset.lower() == 'cho': 
        data = pd.read_csv(r"C:\Users\Byeke\OneDrive\Documents\Spring_2022_Lehigh\data_mining\project_1\Datasets\cho.txt",delimiter='\t',header=None)
    elif dataset.lower() == 'iyer': 
        data = pd.read_csv(r"C:\Users\Byeke\OneDrive\Documents\Spring_2022_Lehigh\data_mining\project_1\Datasets\iyer.txt",delimiter='\t',header=None)
    elif dataset.lower() == 'elliptical': 
        data = pd.read_csv(r"C:\Users\Byeke\OneDrive\Documents\Spring_2022_Lehigh\data_mining\project_1\Datasets\elliptical.txt",delimiter=' ',names = ['x','y'])
    elif dataset.lower() == 'square':
        data = pd.read_csv(r"C:\Users\Byeke\OneDrive\Documents\Spring_2022_Lehigh\data_mining\project_1\Datasets\square.txt",delimiter=' ',names = ['x','y'])
    if dataset.lower() in ['cho','iyer']:
        if cluster_type.lower() == 'kmeans':
            cluster_result,kmeans_centroids,data= real_world_kmeans(data,dataset,remove_outliers=remove_outliers)
            print(data.head())
            data['spectral_label'] = cluster_result
            ##### Do some processing here to get sse calc and accuracy

            accuracy = accuracy_score(cluster_result,data.ground_truth)
            sse_value = sse_calc(data)
            print(f'The accuracy: {accuracy}')
            print(f'The SSE: {sse_value}')
        elif cluster_type.lower() == 'spectral':
            cluster_result,kmeans_centroids,data= real_world_spectral(data,data_name = dataset,remove_outliers = remove_outliers, tol=.001, normalization=False, similarity='gaussian', init_random=True,
                                                                sigma = 2,epsilon =.5, match_truth_values=True)
        
        ##### Do some processing here to get sse calc and accuracy
        data['spectral_label'] = cluster_result
        accuracy = accuracy_score(cluster_result,data.ground_truth)
        sse_value = sse_calc(data)
        print(f'The accuracy: {accuracy}')
        print(f'The SSE: {sse_value}')
    elif dataset.lower() in ['square','elliptical']:
        if cluster_type.lower() == 'kmeans':
            cluster, centroids_current =kmeans_clustering(data.values,tol=.001,k=2,init_random=True,match_truth_values=False, truth_values= [])
            plot_2d_clusters(data,cluster,centroids_current,cluster_type)
            print(cluster, centroids_current)
        elif cluster_type.lower() == 'spectral':
            cluster, centroids_current = generate_spectral_cluster(dataset,k=2,random_init=True,sigma=1,similarity='gaussian',epsilon =.75,normalization = False)
            plot_2d_clusters(data,cluster,centroids_current,cluster_type)
            print(cluster, centroids_current)
    pass

if __name__ == '__main__':
    dataset =  input("Please input dataset: 'Iyer', 'Cho', 'Square', or 'Elliptical'")
    cluster_type = input("Please input clusterting type: 'Kmeans' or 'Spectral'")
    if dataset.lower() == 'iyer':
        remove_outliers = bool(input("Please input 'True' is you want outliers removed, else 'False'"))
        main(dataset,cluster_type,remove_outliers=remove_outliers)
    else:
        main(dataset,cluster_type,remove_outliers=False)

