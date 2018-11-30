from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
import os
import numpy as np

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
data = load_train_data("db/db1/train.txt")
means = calc(data)
for key in data.keys():
    print("Class : ", key)
    print("Train data : ")
    for sample in data[key]:
        print("\t", sample)


#SVM Tree declaration
class Node(object):
    def __init__(self, classes):        
        self.classes = classes                
        self.svm = LinearSVC(C=1000)        
        self.create_left_and_right_child()
        self.train_svm()

    def create_left_and_right_child(self):
        global data, mean
        if len(self.classes) == 1:
            self.left, self.right = None, None
        else:
            kmeans = KMeans(n_clusters=2)    
            X = []
            for k in self.classes:
                X.append(mean[k])
            kmeans.fit(X)
            left_classes = []
            right_classes = []
            for k in self.classes:
                g = kmeans.predict(mean[k])
                if g == 0:
                    left_classes.append(k)
                else:
                    right_classes.append(k)
            self.left = Node(left_classes)
            self.right = Node(right_classes)
    
    def train_svm(self):
        global data
        X, Y = [], [] 
        for k in self.left.classes:
            X.extend(data[k])
            Y.extend(len(data[k]) * [0])
        for k in self.right.classes:
            X.extend(data[k])
            Y.extend(len(data[k]) * [1])
        print("[+] Training SVM")
        self.svm.fit(X,Y)

class SVMTree(object):
    def __init__(self, data):
        self.root = Node(data.keys())
        print("[+] Bilding SVM Tree...")
        self.build_tree(self.root)
        print("[+] Finished")

    def build_tree(self, node):
        if node == None:
            return
        else:            
            node.left.create_left_and_right_child()
            self.build_tree(node.left)
            node.right.create_left_and_right_child()
            self.build_tree(node.right)
    
    def traversal(self, node, x):
        if len(node.classes) == 1:
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
        for x in X:
            result.append(self.traversal(self.root, x))            
        return result


if __name__ == '__main__':
    pass    
            
              