"""
Artificial Neural Network
- Input layer with 2 neurons
- First hidden layer with 4 neurons
- Second hidden layer with 2 neurons
- Output layer of 1 neuron

Note: this file is independent of Flask
"""
import numpy as np 
n2=4
n3=2
def sigmoid(x, derivate = False):
    if derivate:
        return np.exp(-x)/((np.exp(-x)+1)**2)
    else:
        return 1/(1+np.exp(-x))
def mse(yt, ye, derivate=False):
    if derivate:
        return (yt - ye)
    else:
        return np.mean((yt - ye)**2)
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([[1], [0], [0], [1]])
wbdate = {}
np.random.seed(1)
wbdate['W1'] = 2*np.random.random((2,n2)) - 1
wbdate['W2'] = 2*np.random.random((n2,n3)) - 1
wbdate['W3'] = 2*np.random.random((n3,1)) - 1
wbdate['b1'] = 2*np.random.random((1,n2)) - 1
wbdate['b2'] = 2*np.random.random((1,n3)) - 1
wbdate['b3'] = 2*np.random.random((1,1)) - 1
ler=1
for i in range(60000):
    wbdate['L0'] = inputs
    wbdate['L1'] = sigmoid(np.dot(wbdate['L0'], wbdate['W1']) + wbdate['b1'])
    wbdate['L2'] = sigmoid(np.dot(wbdate['L1'], wbdate['W2']) + wbdate['b2'])
    wbdate['L3'] = sigmoid(np.dot(wbdate['L2'], wbdate['W3']) + wbdate['b3'])
    error = mse(outputs, wbdate['L3'], True)
    wbdate['dL3'] = error*sigmoid(wbdate['L3'], derivate = True)
    wbdate['dL2'] = wbdate['dL3'].dot(wbdate['W3'].T)*sigmoid(wbdate['L2'], derivate = True)
    wbdate['dL1'] = wbdate['dL2'].dot(wbdate['W2'].T)*sigmoid(wbdate['L1'], derivate = True)
    wbdate['W3'] += ler*wbdate['L2'].T.dot(wbdate['dL3'])
    wbdate['W2'] += ler*wbdate['L1'].T.dot(wbdate['dL2'])
    wbdate['W1'] += ler*wbdate['L0'].T.dot(wbdate['dL1'])
    wbdate['b3'] += ler*np.sum(wbdate['dL3'], axis=0, keepdims=True)
    wbdate['b2'] += ler*np.sum(wbdate['dL2'], axis=0, keepdims=True)
    wbdate['b1'] += ler*np.sum(wbdate['dL1'], axis=0, keepdims=True)
# inputsx = np.array([[0,0], [0,1], [1,1], [1,0]])
inputsx = np.array([[0,1]])
wbdate['L1'] = sigmoid(np.dot(inputsx, wbdate['W1']) + wbdate['b1'])
wbdate['L2'] = sigmoid(np.dot(wbdate['L1'], wbdate['W2']) + wbdate['b2'])
predictions = sigmoid(np.dot(wbdate['L2'], wbdate['W3']) + wbdate['b3'])
predictions = [round(x[0]) for x in predictions]
print(predictions[0])
accuracy = (1 - np.mean(np.abs(error))) * 100
print("Exact accuracy: " + str(accuracy) + "%")
