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
    inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
    outputs = np.array([[1], [0], [0], [1]])
    np.random.seed(1)
    weights1 = 2*np.random.random((2,n2)) - 1
    weights2 = 2*np.random.random((n2,n3)) - 1
    weights3 = 2*np.random.random((n3,1)) - 1
    bias1 = 2*np.random.random((1,n2)) - 1
    bias2 = 2*np.random.random((1,n3)) - 1
    bias3 = 2*np.random.random((1,1)) - 1
    print(50*"_")
    print(" XOR Neural Network training ...")
    print(50*"_")
    for m in range(50001):
        layer1 = sigmoid(np.dot(inputs, weights1) + bias1)
        layer2 = sigmoid(np.dot(layer1, weights2) + bias2)
        layer3 = sigmoid(np.dot(layer2, weights3) + bias3)
        error = outputs - layer3
        delta_layer3 = error * sigmoid(layer3, derivate = True)
        error_layer2 = delta_layer3.dot(weights3.T)
        delta_layer2 = error_layer2 * sigmoid(layer2, derivate = True)
        error_layer1 = delta_layer2.dot(weights2.T)
        delta_layer1 = error_layer1 * sigmoid(layer1, derivate = True)
        weights3 += layer2.T.dot(delta_layer3)
        weights2 += layer1.T.dot(delta_layer2)
        weights1 += inputs.T.dot(delta_layer1)
        bias3 += np.sum(delta_layer3, axis=0, keepdims=True)
        bias2 += np.sum(delta_layer2, axis=0, keepdims=True)
        bias1 += np.sum(delta_layer1, axis=0, keepdims=True)
    inputsx = np.array([[x1,x2]])
    layer1 = sigmoid(np.dot(inputsx, weights1) + bias1)
    layer2 = sigmoid(np.dot(layer1, weights2) + bias2)
    predictions = sigmoid(np.dot(layer2, weights3) + bias3)
    print(f"Error: {np.mean(np.abs(error))}")
    predictions = [round(x[0]) for x in predictions]
    print(f"Predictions: {predictions}")
    accuracy = (1 - np.mean(np.abs(error))) * 100
    # print(f"Exact accuracy: {accuracy}%")
    return render_template('output.html', result=predictions[0], x1=x1, x2=x2, accuracy=round(accuracy))