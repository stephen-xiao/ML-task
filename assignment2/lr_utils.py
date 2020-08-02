import numpy as np
import h5py
import matplotlib.pyplot as plt
    
def load_dataset():
    train_dataset = h5py.File(('datasets/train_catvnoncat.h5'), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(('datasets/test_catvnoncat.h5'), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
class Logistic():
    def __init__(self,X,Y,test_X,test_Y):
        #
        self.X = X
        self.Y = Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.w = np.zeros((self.n,1))
        self.b = 0
        self.test_Y_pre = np.zeros((1,self.m))

    def sigmoid(self,z):
        #sigmoid
        return 1/(1+np.exp(-z))
    
    def loss(self,Y_hat):
        #
        return (1/self.m)*(-np.sum(self.Y*np.log(Y_hat)+(1-self.Y)*np.log(1-Y_hat)))
    def optimize(self,Y_hat,lr=0.001):
        dz = self.sigmoid(np.dot(self.w.T,self.X) + self.b)-self.Y
        dw = (1/self.m)*np.dot(self.X,dz.T)
        db = (1/self.m)*np.sum(dz)
        self.w -= lr*dw
        self.b -= lr*db
    def model(self,lr=0.001,number_iteration=10000,print_loss=False):
        self.losses = []
        Y_hat = np.zeros((1,self.m))
        for i in range(number_iteration):
            Y_hat = self.sigmoid(np.dot(self.w.T,self.X)+self.b)
            loss = self.loss(Y_hat)
            self.optimize(Y_hat,lr)
            if i%100==0:
                print('iteration:{:d},the loss is {:.6f}'.format(i,loss))
                if print_loss==True:
                    self.losses.append(loss)
        Y_pre = np.zeros_like(Y_hat)
        Y_pre[Y_hat > 0.5]= 1 
        Y_pre[Y_hat <= 0.5]= 0
        accuracy = 1-(1/self.m)*np.sum(np.abs(self.Y-Y_pre))
        print('The number of iteration is {:d} and the accuracy of train is {:.6f}%'.format(number_iteration,accuracy*100))
    def predict(self):
        Y_pre = np.dot(self.w.T,self.test_X) + self.b
        Y_pre[Y_pre > 0.5]= 1 
        Y_pre[Y_pre <= 0.5]= 0
        accuracy = 1-(1/self.m)*np.sum(np.abs(self.test_Y-Y_pre))
        print('The accuracy of test is {:.6f}%'.format(accuracy*100))
        
if __name__ == '__main__':
    train_data,train_labels,test_data,test_labels,classes = load_dataset()
    train_data = train_data.reshape(train_data.shape[0],-1).T/255
    test_data = test_data.reshape(test_data.shape[0],-1).T/255
    logistic = Logistic(train_data,train_labels,test_data,test_labels)
    logistic.model(lr=0.01,number_iteration=10000,print_loss=True)
    logistic.predict()
    plt.plot(logistic.losses)
    plt.show()
    plt.xlabel('number_iteration')
    plt.ylabel('loss')
    plt.title('Logistic')
