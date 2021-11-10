import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import mean_squared_error
from utils import compute_average_accuracy


class GD:
    def __init__(self, max_iter, num_features, eta, lam):
        super().__init__()
        self.max_iter = max_iter
        self.eta = eta
        self.lam = lam
        self.num_features = num_features
        self.weights = None
        self.bias = None

     # TODO: complete implementation
    def init_parameters(self):
        # complete implementation
        self.weights=np.zeros(self.num_features) #initialize weight
        self.bias=0     #initialise bias


    # TODO: complete implementation
    def train(self, train_X, train_y, dev_X, dev_y,):
        self.init_parameters()
        # complete implementation
        train_acc=[] #create empty list
        dev_acc=[]  #create empty list
        train_loss=[]   #create empty list
        dev_loss=[]     #create empty list

        for iter in range(self.max_iter):
            predicted_labels=[] #create empty list
            g_b=0   #initialize derivative of bias
            g_w=np.zeros(self.num_features) #initialise derivative of weights
            f_w=0 #initialise loss function
            d_predicted_labels=[]
            for Xx,Yy in zip(train_X,train_y):
                activation=safe_sparse_dot(self.weights,Xx.transpose())+self.bias       #wx+b
                f_w+= (activation-Yy)**2            #loss function
                g_w += 2*(activation-Yy)*Xx         #update derivative of weights
                g_b += 2*(activation-Yy)            #update derivative of bias

            train_avg_loss=f_w/(len(train_y))       #average loss
            g_w += (self.lam * self.weights)        #update derivative of weights with lambda*weights
            self.weights -= (self.eta*g_w)          #update weights
            self.bias -= (self.eta*g_b)             #update bias
            for Xx,Yy in zip(train_X,train_y):
                activation=safe_sparse_dot(self.weights,Xx.transpose())+self.bias   #wx+b
                if activation > 0: #predicts label
                    y_hat= 1    
                else:
                    y_hat= -1
                predicted_labels.append(y_hat) #store predicted label
            df_w=0 #initialise dev loss function
            for dX,dy in zip(dev_X,dev_y):
                d_activation=safe_sparse_dot(self.weights,dX.transpose())+self.bias #wx+b
                df_w+= (d_activation-dy)**2 #loss function
                if d_activation > 0: #predicts label
                    d_hat= 1
                else:
                    d_hat= -1
                d_predicted_labels.append(d_hat)
            dev_avg_loss=df_w/dev_X.shape[0] #average dev loss
            
            t_acc=compute_average_accuracy(predicted_labels,train_y) #train accuracy
            d_acc=compute_average_accuracy(d_predicted_labels,dev_y) #dev accuracy
            train_acc.append(t_acc)
            dev_acc.append(d_acc)   
            train_loss.append(train_avg_loss)
            dev_loss.append(dev_avg_loss)
            print("\nIteration ", iter,"\nTrain Accuracy",t_acc,"\nTrain Average Loss",train_avg_loss,"\nDev Accuracy",d_acc,"\nDev Average Loss",dev_avg_loss)
        return train_acc, dev_acc,train_loss,dev_loss


    # TODO: complete implementation
    def predict(self, X, y=None):
        predicted_labels = []
        pred_avg_loss = 0.0
        f_w=0 #initialise loss function
        for i,j in zip(X,y):
            activation=safe_sparse_dot(self.weights,i.transpose())+self.bias #wx+b
            f_w+=(activation-j)**2 #loss function
            if activation == 0: #predicts label
                y_hat= -1
            else:
                y_hat= np.sign(activation)
            predicted_labels.append(y_hat)
        
        pred_avg_loss=f_w/X.shape[0]    #average loss
        # complete implementation
        return predicted_labels, pred_avg_loss
