from matplotlib import pyplot as plt
import numpy as np

data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],  # data from the youtube series
        [3.5, 0.5, 1],
        [2, 0.5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

mystery_flower = [3,1.5]  # data from the youtube series that the farmer forgot to record the color of the flower

# NEURAL NETWORK

# since we are looking at predicting the color of the flower, red = 1 and blue = 0, we can use sigmoid as the activation function

def sigmoid(x):  # this is the function that we need to minimize
    return (1 / (1 + np.exp(-x)))


def sigmoid_p(x):  # derivative of the sigmoid function
    return sigmoid(x) * (1 - 1 * sigmoid(x))


# training loop to feed the data and minimize the function

learning_rate = 0.01
costs = []


w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn() * 0.001


for i in range(40000):
    ri = np.random.randint(len(data))  # this command spits out random indices for data array
    # print (ri)
    point = data[ri]  # random data points from data array

    # print(point) - you are randomly calling the points from the 'data' set defined earlier. This is because you have already defined 'ri' which is randomly calling indices

    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)

    target = point[2]

    cost = (pred - target) ** 2  # cost function - we need to minimize this
    dcost_pred = 2*(pred-target)
    dpred_dz = sigmoid_p(z)

    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dz = dcost_pred * dpred_dz

    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db

    if i % 100 == 0:
        cost_sum = 0
        for j in range (len(data)):
            point = data[ri]

            z = point [0]*w1 + point [1]*w2 + b
            pred = sigmoid(z)
            target = point [2]
            cost_sum += np.square(pred - target)

        costs.append(cost_sum/len(data))
plt.plot(costs)
# plt.show()

# print predictions

for i in range (len(data)):
    point = data[i]
    # print (point)
    z = point [0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)
    # print ("pred: {}".format(pred))

# mystery flower

# z = mystery_flower[0] * w1 + mystery_flower[1] * w2 + b
# pred = sigmoid(z)
# print (pred)

from gtts import gTTS
import os

def which_flower (length, width):
    z = length* w1 + width + b
    pred = sigmoid(z)
    print (pred)

    # if pred < 0.5:
        # tts = gTTS(text = "blue", lang ='en')
        # tts.save("pcvoice.mp3")
        # os.system("start pcvoice.mp3")
    # else:
        # tts = gTTs(text="red", lang='en')
        # tts.save("pcvoice.mp3")
        # os.system("start pcvoice.mp3")


which_flower(6,6)