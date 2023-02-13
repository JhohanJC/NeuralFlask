from network import app 
from flask import render_template, request
import numpy as np 
@app.route('/')
@app.route('/inicio')
def index():
    return render_template('index.html')

@app.route('/training_predict', methods=['POST'])
def training_predict():
    try:
        x1 = int(request.form.get('x1'))
        x2 = int(request.form.get('x2'))
    except ValueError as ex:
        return f"¡Incorrect!. Please enter a binary number"
    x1 = int(request.form['x1'])
    x2 = int(request.form['x2'])
    n2=4 # 4 neurons in the first hidden layer
    n3=2 # 2 neurons in the second hidden layer
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
    # np.random.seed(1)
    # weights1 = 2*np.random.random((2,n2)) - 1
    # weights2 = 2*np.random.random((n2,n3)) - 1
    # weights3 = 2*np.random.random((n3,1)) - 1
    # bias1 = 2*np.random.random((1,n2)) - 1
    # bias2 = 2*np.random.random((1,n3)) - 1
    # bias3 = 2*np.random.random((1,1)) - 1
    wbdate = {}
    np.random.seed(1)
    wbdate['W1'] = 2*np.random.random((2,n2)) - 1
    wbdate['W2'] = 2*np.random.random((n2,n3)) - 1
    wbdate['W3'] = 2*np.random.random((n3,1)) - 1
    wbdate['b1'] = 2*np.random.random((1,n2)) - 1
    wbdate['b2'] = 2*np.random.random((1,n3)) - 1
    wbdate['b3'] = 2*np.random.random((1,1)) - 1
    print(50*"_")
    print(" XOR Neural Network training ...")
    print(50*"_")
    ler=1
    for m in range(50001):
        # layer1 = sigmoid(np.dot(inputs, weights1) + bias1)
        # layer2 = sigmoid(np.dot(layer1, weights2) + bias2)
        # layer3 = sigmoid(np.dot(layer2, weights3) + bias3)
        wbdate['L0'] = inputs
        wbdate['L1'] = sigmoid(np.dot(wbdate['L0'], wbdate['W1']) + wbdate['b1'])
        wbdate['L2'] = sigmoid(np.dot(wbdate['L1'], wbdate['W2']) + wbdate['b2'])
        wbdate['L3'] = sigmoid(np.dot(wbdate['L2'], wbdate['W3']) + wbdate['b3'])
        # error = outputs - layer3
        # delta_layer3 = error * sigmoid(layer3, derivate = True)
        # error_layer2 = delta_layer3.dot(weights3.T)
        # delta_layer2 = error_layer2 * sigmoid(layer2, derivate = True)
        # error_layer1 = delta_layer2.dot(weights2.T)
        # delta_layer1 = error_layer1 * sigmoid(layer1, derivate = True)
        error = mse(outputs, wbdate['L3'], True)
        wbdate['dL3'] = error*sigmoid(wbdate['L3'], derivate = True)
        wbdate['dL2'] = wbdate['dL3'].dot(wbdate['W3'].T)*sigmoid(wbdate['L2'], derivate = True)
        wbdate['dL1'] = wbdate['dL2'].dot(wbdate['W2'].T)*sigmoid(wbdate['L1'], derivate = True)
        # weights3 += layer2.T.dot(delta_layer3)
        # weights2 += layer1.T.dot(delta_layer2)
        # weights1 += inputs.T.dot(delta_layer1)
        wbdate['W3'] += ler*wbdate['L2'].T.dot(wbdate['dL3'])
        wbdate['W2'] += ler*wbdate['L1'].T.dot(wbdate['dL2'])
        wbdate['W1'] += ler*wbdate['L0'].T.dot(wbdate['dL1'])
        # bias3 += np.sum(delta_layer3, axis=0, keepdims=True)
        # bias2 += np.sum(delta_layer2, axis=0, keepdims=True)
        # bias1 += np.sum(delta_layer1, axis=0, keepdims=True)
        wbdate['b3'] += ler*np.sum(wbdate['dL3'], axis=0, keepdims=True)
        wbdate['b2'] += ler*np.sum(wbdate['dL2'], axis=0, keepdims=True)
        wbdate['b1'] += ler*np.sum(wbdate['dL1'], axis=0, keepdims=True)
    inputsx = np.array([[x1,x2]])
    # layer1 = sigmoid(np.dot(inputsx, weights1) + bias1)
    # layer2 = sigmoid(np.dot(layer1, weights2) + bias2)
    # predictions = sigmoid(np.dot(layer2, weights3) + bias3)
    wbdate['L1'] = sigmoid(np.dot(inputsx, wbdate['W1']) + wbdate['b1'])
    wbdate['L2'] = sigmoid(np.dot(wbdate['L1'], wbdate['W2']) + wbdate['b2'])
    predictions = sigmoid(np.dot(wbdate['L2'], wbdate['W3']) + wbdate['b3'])
    print(f"Error: {np.mean(np.abs(error))}")
    predictions = [round(x[0]) for x in predictions]
    print(f"Predictions: {predictions}")
    accuracy = (1 - np.mean(np.abs(error))) * 100
    # print(f"Exact accuracy: {accuracy}%")
    return render_template('output.html', result=predictions[0], x1=x1, x2=x2, accuracy=round(accuracy))