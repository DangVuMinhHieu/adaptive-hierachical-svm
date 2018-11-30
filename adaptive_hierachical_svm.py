from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
import os
import numpy as np
import pickle
import sys
def load_train_data(db_train):
    temp = {}
    print("[+] Load train data....")
    with open(db_train, "r") as f:
        for file in f:
            file = file[:-1]
            _, label = file.split("/")[1].split(".")  
            file = file.replace("images","features/vgg16_fc2").replace(".jpg", ".npy")          
            #labels.append(label)
            if label not in temp:
                temp[label] = []
            temp[label].append(np.load(file)[0])                    
        return temp

def calc(data):
    temp = {}
    for k in data.keys():
        temp[k] = np.average(data[k], axis=0)
    return temp

#Create global data & mean of each classes
data = None  #Change train file here
means = None

#SVM Tree declaration
class Node(object):
    def __init__(self, classes):        
        self.classes = classes                
        self.svm = LinearSVC(C=1000)
    
    def build_node(self):
        global data, means
        if self.is_leaf():
            self.left = None
            self.right = None
        else:
            kmeans = KMeans(n_clusters=2)    
            X = []
            for k in self.classes:
                X.append(means[k].tolist())
            kmeans.fit(X)
            left_classes = []
            right_classes = []
            for k in self.classes:
                g = kmeans.predict([means[k]])
                if g == 0:
                    left_classes.append(k)
                else:
                    right_classes.append(k)
            self.left = Node(left_classes)
            self.right = Node(right_classes)
            self.train_svm()
                    
    def is_leaf(self):
        return  len(self.classes) == 1

    def train_svm(self):
        global data         
        X = []
        Y = []    
        for k in self.left.classes:
            X.extend(data[k])
            Y.extend(len(data[k]) * [0])
        for k in self.right.classes:
            X.extend(data[k])
            Y.extend(len(data[k]) * [1])        
        self.svm.fit(X,Y)        

class SVMTree(object):
    def __init__(self, data):
        print("[+] Building SVM Tree...")
        self.root = Node(list(data.keys()))
        self.build_tree(self.root)
        print("[+] Finished building svm tree")

    def build_tree(self, node):
        if node == None:
            return
        else:            
            node.build_node()
            self.build_tree(node.left)            
            self.build_tree(node.right)
    
    def traversal(self, node, x):
        if node.is_leaf():
            return list(node.classes)[0]
        else:
            pred = node.svm.predict(x)
            if pred == 0:
                return self.traversal(node.left, x)
            else:
                return self.traversal(node.right, x)

    def predict(self, X):
        result = []
        if not isinstance(X, list):
            X = list(X)        
            result.append(self.traversal(self.root, X))            
        return result


if __name__ == '__main__':    
    data = load_train_data(sys.argv[1])  #Change train file here
    means = calc(data)
    tree = SVMTree(data)
    with open("models/tree_svm.pkl", "wb") as f:
        print("[+] Saving svm tree to file...")
        pickle.dump(tree, f, -1)
        print("[+] Finished")
            
              
