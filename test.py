from sklearn.cluster import KMeans
from sklearn.externals import joblib
from features_extractor import extract_features, save_features
from sklearn.metrics import accuracy_score
import numpy as np
import os
import pickle
from adaptive_hierachical_svm import SVMTree, Node
import random
import sys

def extract_features_for_test(db="db/db1/test.txt"):
    with open(db, "r") as file:
        for i,line in enumerate(file):
            img_path = line[:-1]
            print("[+] Read image  : ", img_path," id : ", i)
            if os.path.isfile(img_path) and img_path.find(".jpg") != -1:            
                save_path = img_path.replace("images", "features/vgg16_fc2/test").replace(".jpg", ".npy")
                save_path = save_path.split("/")
                save_path[-2] = save_path[-2] + "+" + save_path[-1]
                save_path = '/'.join(save_path[:-1])
                print("[+] Extract feature from image : ", img_path)
                features = extract_features(img_path)
                save_features(save_path, features)

if __name__ == "__main__":
    #extract_features_for_test()    
    print("[+] Load svm tree :")
    f = open("models/tree_svm.pkl", "rb")
    tree = pickle.load(f)
    f.close()
    print("[+] Finised")

    gts = []
    preds = []
    tests = []
    print("[+] Loading test data....")
    for file in os.listdir(sys.argv[1]):
        tests.append(file)      
    random.shuffle(tests)        
    print("[+] Finished")
    print("[+] Testing....")
    for test in tests:       
        features = np.load(os.path.join(sys.argv[1], test))
        gts.append(test.split('+')[0].split('.')[1])
        pred = tree.predict(features)
        preds.extend(pred)
        print("Test : ", test, " | pred : ", pred)    
    print("[+] Accuracy : ", accuracy_score(preds, gts))        
