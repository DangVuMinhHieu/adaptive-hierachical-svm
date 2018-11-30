from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
import os
import sys
def load_features(src):
    print("[+] Load data....")
    data = []
    for folder in os.listdir(src):
        folder_path = os.path.join(src, folder)        
        for file in os.listdir(folder_path):
            data.append(np.load(os.path.join(folder_path, file))[0])
    print("[+] Load data finished")
    return data

def clustering(data):
    print("[!] Clustering data...")
    kmeans = KMeans(n_clusters=256, max_iter=10, random_state=0).fit(data)
    print("[+] Finished")
    return kmeans

def save_model(model, name):
    file_name = name + ".joblib"
    print("[+] Saving model to file : " ,file_name)
    joblib.dump(model, file_name)

if __name__=='__main__':
    src = sys.argv[1]
    data = load_features(src)
    kmeans = clustering(data)
    save_model(kmeans, "models/k_means")
