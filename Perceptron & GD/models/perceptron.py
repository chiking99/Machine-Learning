import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
import random
from utils import compute_average_accuracy 

class Perceptron():
    def __init__(self, num_epochs, num_features, averaged):
        super().__init__()
        self.num_epochs = num_epochs
        self.averaged = averaged
        self.num_features = num_features
        self.weights = None
        self.bias = None

    # TODO: complete implementation
    #https://weimin17.github.io/2017/09/Implement-The-Perceptron-Algorithm-in-Python-version2/
    def init_parameters(self):
        # complete implementation
        self.bias=0 #initialize bias
        self.weights=np.zeros(self.num_features) #initialize weights


    # TODO: complete implementation
    def train(self, train_X, train_y, dev_X, dev_y,shuffle):
        self.init_parameters()
        # complete implementation
        train_acc=[]    #create empty list
        dev_acc=[]      #create empty list
        for epoch in range(self.num_epochs):
            #https://stackoverflow.com/questions/59033298/how-to-shuffle-the-training-data-set-for-each-epochs-while-mantaining-the-intial
            predicted_labels = []   #create empty list
            d_predicted_labels=[]   #create empty list
            iter_=np.arange(train_X.shape[0])   
            if shuffle==True:   #shuffle
                np.random.shuffle(iter_) #shuffle the list
                train_X_copy = train_X.copy() #copy the train dataset
                train_y_copy = train_y.copy()
                train_X=train_X_copy[iter_]    #shuffled train dataset
                train_y=train_y_copy[iter_]
            else:
                train_X,train_y=train_X,train_y 
            for j in range(train_X.shape[0]) :  
                activation = safe_sparse_dot(self.weights,train_X[j,:].transpose()) + self.bias    #wx+b
                if activation > 0: #predict labels
                    y_hat= 1
                else:
                    y_hat= -1
                predicted_labels.append(y_hat) #store predicted labels
                if train_y[j]*activation <= 0:  #update weight and bias
                    self.weights += train_y[j]*train_X[j,:]
                    self.bias += train_y[j]
            for Dx in dev_X :
                d_activation=safe_sparse_dot(self.weights,Dx.transpose()) + self.bias #wx+b
                if d_activation > 0: #predict labels
                    d_hat= 1
                else:
                    d_hat= -1
                d_predicted_labels.append(d_hat)
            t_acc=compute_average_accuracy(predicted_labels,train_y) #train accuracy
            d_acc=compute_average_accuracy(d_predicted_labels,dev_y) #dev accuracy
            train_acc.append(t_acc)
            dev_acc.append(d_acc)
            print("\nEpoch ", epoch,"\nTraining accuracy",t_acc,"\nDev accuracy",d_acc)
        return train_acc, dev_acc


    # TODO: complete implementation
    def predict(self, X):
        predicted_labels = [] #create empty list
        # complete implementation
        #https://learnai1.home.blog/2019/11/16/perceptron-delta-rule-python-implementation/
        for x in X:
            activation = safe_sparse_dot(self.weights,x.transpose()) + self.bias #wx+b
            if activation > 0: #predict labels
                y_hat= 1
            else:
                y_hat= -1
            predicted_labels.append(y_hat)

        return predicted_labels

