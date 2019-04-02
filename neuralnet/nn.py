import numpy as np
import matplotlib.pyplot as plt

# Load data

def get_data(file):
    data = open(file).readlines()
    x_train, y_train = [], []
    for datum in data:
        split = [int(i) for i in datum.strip().split()]
        x_train.append(np.array(split[:-1]))
        y_train.append(np.array(split[-1]))
        print(len(split))
        print(len(x_train[0]))
    x_train, y_train = np.array(x_train), np.array(y_train)

    # One hot encode y_train
    _ = np.zeros((len(y_train), 10))
    _[np.arange(len(y_train)), y_train] = 1
    y_train = _
    
    return x_train, y_train

x_train, y_train = get_data("digits-training.data")
    

in_size = 64
hid_size = 32
out_size = 10

#in_size = 1
#hid_size = 10
#out_size = 1

# Training data
#np.random.seed(1)
#x_train = np.linspace(-1, 1, 200)[:, None]       # [batch, 1]
#y_train = x_train ** 2                                  # [batch, 1]
learning_rate = 0.001

# Helpers
def sigmoid(x):
    return np.tanh(x)

def derivative_sigmoid(x):
    return 1 - np.tanh(x)**2
# End helpers


# Weights for layer 1 (input layer)
w1 = np.random.uniform(-0.1, 0.1, (in_size, hid_size))
# Weights for layer 2 (hidden layer)
w2 = np.random.uniform(-0.1, 0.1, (hid_size, hid_size))
# Weights for layer 2 (hidden layer)
w3 = np.random.uniform(-0.1, 0.1, (hid_size, hid_size))
# Weights for layer 2 (hidden layer)
w4 = np.random.uniform(-0.1, 0.1, (hid_size, hid_size))
# Weights for layer 3 (output)
w5 = np.random.uniform(-0.1, 0.1, (hid_size, out_size))

print(x_train.shape)

batch_size = 32

# Biases
b1 = np.full((batch_size, hid_size), 0.1)
b2 = np.full((batch_size, hid_size), 0.1)
b3 = np.full((batch_size, hid_size), 0.1)
b4 = np.full((batch_size, hid_size), 0.1)
b5 = np.full((batch_size, out_size), 0.1)

print(x_train.shape)
    
for i in range(100000):
    randindex = np.random.randint(0, len(x_train)-batch_size)
    x = x_train[randindex:randindex+32, :]
    y = y_train[randindex:randindex+32, :]

    a1 = x
    z2 = a1.dot(w1) + b1
    a2 = sigmoid(z2)
    z3 = a2.dot(w2) + b2
    a3 = sigmoid(z3)
    z4 = a3.dot(w3) + b3
    a4 = sigmoid(z4)
    z5 = a4.dot(w4) + b4
    a5 = sigmoid(z5)
    z6 = a5.dot(w5) + b5
    z6 = sigmoid(z6)
    
    cost = np.sum((z6 - y)**2)/2

    # backpropagation
    z6_delta = z6 - y
    
    dw5 = a5.T.dot(z6_delta)
    db5 = np.sum(z6_delta, axis=0, keepdims=True)

    z5_delta = z6_delta.dot(w5.T) * derivative_sigmoid(z5)
    dw4 = a4.T.dot(z5_delta)
    db4 = np.sum(z5_delta, axis=0, keepdims=True)

    z4_delta = z5_delta.dot(w4.T) * derivative_sigmoid(z4)
    dw3 = a3.T.dot(z4_delta)
    db3 = np.sum(z4_delta, axis=0, keepdims=True)

    z3_delta = z4_delta.dot(w3.T) * derivative_sigmoid(z3)
    dw2 = a2.T.dot(z3_delta)
    db2 = np.sum(z3_delta, axis=0, keepdims=True)

    z2_delta = z3_delta.dot(w2.T) * derivative_sigmoid(z2)
    dw1 = x.T.dot(z2_delta)
    db1 = np.sum(z2_delta, axis=0, keepdims=True)

    
    # update parameters
    for param, gradient in zip([w1, w2, w3, w4, w5, b1, b2, b3, b4, b5],
                               [dw1, dw2, dw3, dw4, dw5, db1, db2, db3, db4, db5]):
    
        param -= learning_rate * gradient

    print(cost)

# test it

cost_total = 0
total = 0
correct = 0
x_test, y_test = get_data("digits-test.data")
for i in range(len(x_test)%32):
    x = x_test[i*32:32*i+32, :]
    y = y_test[i*32:32*i+32, :]

    a1 = x
    z2 = a1.dot(w1) + b1
    a2 = sigmoid(z2)
    z3 = a2.dot(w2) + b2
    a3 = sigmoid(z3)
    z4 = a3.dot(w3) + b3
    a4 = sigmoid(z4)
    z5 = a4.dot(w4) + b4
    a5 = sigmoid(z5)
    z6 = a5.dot(w5) + b5
    z6 = sigmoid(z6)

    for z_v, y_v in zip(z6, y):
        prediction = np.argmax(z_v)
        actual = np.argmax(y_v)
        if prediction == actual:
            correct += 1
        total += 1
        
    cost = np.sum((z6 - y)**2)/2

    cost_total += cost
    print(cost)

print(cost_total/len(x_test)%32)
print(correct/total)
