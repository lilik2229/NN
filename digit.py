# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from matplotlib import cm
from sklearn.datasets import load_digits
from numpy import ndarray

digits = load_digits()
X = digits.data
Y = digits.target
nn_input_dim = X.shape[1]
nn_output_dim = 9
num_examples = 25
epsilon = 0.01
reg_lambda = 0.01
y = np.zeros((num_examples,10))

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
        print type(W1[0])
        print type(X[0])
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        delta3 = probs
        print delta3.shape
        delta3[range(num_examples)] -= y[range(num_examples)]
        
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


p = np.random.random_integers(0,len(X),num_examples)
samples = np.array(list(zip(X,Y)))[p]

X = samples[:,0]
X2 = X[0]
for i in X[1:]:
    X2 = np.c_[X2,i]
X = X2.T
print X.shape

for index,(data ,label) in enumerate(samples):
    plt.subplot(5,5,index+1)
    plt.imshow(data.reshape(8, 8), cmap=cm.gray_r, interpolation='nearest')
    plt.axis("off")
    # 画像データのタイトルに正解ラベルを表示する
    y[index,label] = 1
    plt.title(label, color='red')

    
#print y
# グラフを表示する
# plt.show()    

model = build_model(3,print_loss=True)

