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
    inputsx = np.array([[x1,x2]])
    wbdate['L1'] = sigmoid(np.dot(inputsx, wbdate['W1']) + wbdate['b1'])
    wbdate['L2'] = sigmoid(np.dot(wbdate['L1'], wbdate['W2']) + wbdate['b2'])
    predictions = sigmoid(np.dot(wbdate['L2'], wbdate['W3']) + wbdate['b3'])
    print(f"Error: {np.mean(np.abs(error))}")
    predictions = [round(x[0]) for x in predictions]
    print(f"Predictions: {predictions}")
    accuracy = (1 - np.mean(np.abs(error))) * 100
    # print(f"Exact accuracy: {accuracy}%")
    return render_template('output.html', result=predictions[0], x1=x1, x2=x2, accuracy=round(accuracy))