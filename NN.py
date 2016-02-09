# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

np.random.seed(1)
X,y = sklearn.datasets.make_moons(200, noise=0.10)
plt.scatter(X[:,0],X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X,y)


num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2
epsilon = 0.01
reg_lambda = 0.01

def plot_decision_boundary(pred_func):
    x_min, x_max = X[:,0].min() - .5 ,X[:,0].max() + .5
    y_min, y_max = X[:,1].min() - .5 ,X[:,1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))

    Z = pred_func(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

def calculate_loss(model):
    W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)))
    return 1./num_examples * data_loss

def predict(model, x):
    W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs,axis=1)

def build_model(nn_hdim, num_passes=20000 , print_loss=False):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim,nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1,nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1,nn_output_dim))
    model = {}
    
    for i in xrange(0, num_passes):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        delta3 = probs
        delta3[range(num_examples),y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3,axis=0,keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1,2))
        dW1 = np.dot(X.T,delta2)
        db1 = np.sum(delta2,axis=0)

        dW2 +=reg_lambda * W2
        dW1 +=reg_lambda * W1

        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        model = {'W1': W1,'b1': b1,'W2': W2 , 'b2': b2}

        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" %(i, calculate_loss(model))

    return model

model = build_model(3,print_loss=True)

plot_decision_boundary(lambda x: predict(model,x))
plt.title("Decision Boundary for hidden layer size 3")
# np.random.seed(0)
# X,y = sklearn.datasets.make_moons(200, noise=0.20)
# plt.scatter(X[:,0],X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X,y)
# plot_decision_boundary(lambda x: clf.predict(x))


plt.show()



