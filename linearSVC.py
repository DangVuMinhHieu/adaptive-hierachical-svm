from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import numpy as np
import sys
import os

def load_train_data(db):
    X = []
    Y = []
    print("[+] Load train data....")
    with open(db, "r") as f:
        for file in f:
            file = file[:-1]
            _, label = file.split("/")[1].split(".")  
            file = file.replace("images","features/vgg16_fc2").replace(".jpg", ".npy")          
            #labels.append(label)
            X.append(np.load(file)[0])
            Y.append(label)
        return X, Y

def load_test_data(src):
    X =[]
    Y =[]
    print("[+] Load test data....")
    for file in os.listdir(src):
        label = file.split("+")[0]
        X.append(np.load(os.path.join(src, file))[0])
        Y.append(label.split(".")[1])
    return X, Y

if __name__ == "__main__":
    X_train, Y_train = load_train_data("db/db1/train.txt")
    #np.save("train_X.npy", X_train)
    #np.save("train_Y.npy", Y_train)
    print("[+] Training model....")
    clf = LinearSVC(C=1000)
    clf.fit(X_train, Y_train)
    joblib.dump(clf, "svc_model_2.joblib")
    #print("[+] Load svc_model")
    #clf = joblib.load("models/svc_model_2.joblib")
    X_test, Y_test = load_test_data("features/vgg16_fc2/test") #test is a folder contains feautures of testing images
    #np.save("test_X.npy", X_test)
    #np.save("test_Y.npy", Y_test)
    #X_test = np.load("test_X.npy")
    #Y_test = np.load("test_Y.npy")
    print("[|] Predicting.....")
    Y_pred = clf.predict(X_test)
    print("[+] Accuracy : ", accuracy_score(Y_test, Y_pred))
