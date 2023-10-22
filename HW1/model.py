import numpy as np

class linear_Regression():
    def __init__(self) -> None:
        pass

    def cost_function(self, X, y, theta):
        m = len(y)
        y_pred = X.dot(theta)
        error = (y_pred - y) ** 2

        return 1 / (2 * m) * np.sum(error)
    
    def gradient_descent(self, X, y, theta, Learningrate, epochs):
        m = len(y)
        cost=[]
        for i in range(epochs):
            y_pred = X.dot(theta)
            error = np.dot(X.transpose(), (y_pred-y))
            theta -= Learningrate * 1/m * error
            cost.append(self.cost_function(X, y, theta))
        return theta, cost

class logisticRegression():
    def sigmod(self,z):
        return 1/(1+np.exp(-z))
    
    def cost_function(self,X,Y, Theta):
        m = len(Y)
        h = self.sigmod(X.dot(Theta))
        j = 1/m *(Y.T.dot(np.log(h)) + (1-Y).T.dot(np.log(1-h)))
        return j
    
    def gradient (self, X, y, theta, Learningrate, epochs):
        m = len(y)
        cost = []
        for i in range(epochs):
            y_pred = self.sigmod(X.dot(theta))
            gradient = (1/m) * X.T.dot(y_pred-y)
            theta -= Learningrate * gradient
            cost.append(self.cost_function(X,y,theta))
        
        return theta,cost

