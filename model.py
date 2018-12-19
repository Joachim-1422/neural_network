#!/usr/bin/env python3

import numpy as np
import pandas

def sort_data(data):
    n_sex = []
    for i in range(0, len(data.Sex) - 1):
        if data.Sex[i] in ['male']:
            n_sex.append(0)
        else:
            n_sex.append(1)
    embs = []
    embc = []
    embq = []
    for i in range(0, len(data.Embarked) - 1):
        if data.Embarked[i] in ['S']:
            embs.append(1)
            embc.append(0)
            embq.append(0)
        elif data.Embarked[i] in ['C']:
            embs.append(0)
            embc.append(1)
            embq.append(0)
        else:
            embs.append(0)
            embc.append(0)
            embq.append(1)
    pc1 = []
    pc2 = []
    pc3 = []
    for i in range(0, len(data.Pclass) - 1):
        if data.Pclass[i] == 1:
            pc1.append(1)
            pc2.append(0)
            pc3.append(0)
        elif data.Pclass[i] == 2:
            pc1.append(0)
            pc2.append(1)
            pc3.append(0)
        else:
            pc1.append(0)
            pc2.append(0)
            pc3.append(1)
    n_sur = []
    for i in range(0, len(data.Survived) - 1):
        if data.Survived[i] == 0:
            n_sur.append(0)
        else:
            n_sur.append(1)
    fill = data.Age.median()
    data.Age = data.Age.fillna(fill)
    data.Age = data.Age / 100
    data.SibSp = data.SibSp / 8
    data.Parch = data.Parch / 6
    return n_sex, embs, embc, embq, n_sur, pc1, pc2, pc3

def get_data():
    '''
    Get Kaggle's data from train.csv
    '''
    data = pandas.read_csv('all/train.csv')
    sex, embs, embc, embq, survivor, class1, class2, class3 = sort_data(data)
    feature = np.array([class1[0], class2[0], class3[0], sex[0], data.Age[0], data.SibSp[0], data.Parch[0], embs[0], embc[0], embq[0]])
    for i in range(1, len(data) - 1):
        x = feature
        feature = np.vstack((x, [class1[i], class2[i], class3[i], sex[i], data.Age[i], data.SibSp[i], data.Parch[i], embs[i], embc[i], embq[i]]))
    target = np.array(survivor)
    return feature, target

def init_variables():
    weights = np.random.normal(size=(10, 20))
    bias = np.zeros(20)
    w2 = np.random.normal(size=(20))
    b2 = 0
    return weights, w2, bias, b2

def pre_activation(features, weights, bias):
    return np.dot(features, weights) + bias

def activation(z):
    return 1 / (1 + np.exp(-z))

def derivative_activation(z):
    return activation(z) * (1 - activation(z))

def predict(features, weights, bias, w2, b2):
    z = pre_activation(features, weights, bias)
    a1 = activation(z)
    z2 = pre_activation(a1, w2, b2)
    y = activation(z2)
    return np.round(y)

def cost(predictions, targets):
    return np.mean((predictions - targets)**2)

def first_gradients(y, t, z, z2, w2):
    return (y - t) * np.expand_dims(derivative_activation(z2), 0) * w2 * derivative_activation(z)

def nd_gradients(y, t, z):
    return (y - t) * derivative_activation(z)

def train(weights, features, targets, bias, w2, b2):
    epochs = 200
    learning_rate = 0.1
    predictions = predict(features, weights, bias, w2, b2)
    print("Accuracy", np.mean(predictions == targets))
    for epoch in range(epochs):
        if epoch % 10 == 0:
            predictions = predict(features, weights, bias, w2, b2)
            print("Cost = %s" % cost(predictions, targets))
        weights_gradients = np.zeros(weights.shape)
        bias_gradients = np.zeros(bias.shape)
        w2_gradients = np.zeros(w2.shape)
        b2_gradients = 0
        for feature, target in zip(features, targets):
            z = pre_activation(feature, weights, bias)
            a1 = activation(z)
            z2 = pre_activation(a1, w2, b2)
            y = activation(z2)
            weights_gradients += first_gradients(a1, target, z, z2, w2) * feature[:,None]
            w2_gradients += nd_gradients(y, target, z2) * a1
            bias_gradients += first_gradients(a1, target, z, z2, w2)
            b2_gradients += nd_gradients(y, target, z2)
        weights = weights - learning_rate * weights_gradients
        w2 = w2 - learning_rate * w2_gradients
        bias = bias - learning_rate * bias_gradients
        b2 = b2 - learning_rate * b2_gradients
    predictions = predict(features, weights, bias, w2, b2)
    print("Accuracy", np.mean(predictions == targets))

def main():
    features, targets = get_data()
    weights, w2, bias, b2 = init_variables()
    train(weights, features, targets, bias, w2, b2)
    return

if __name__ == '__main__':
    main()