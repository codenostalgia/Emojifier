import numpy as np
from emo_utils import *
import pickle


file_name = "emojifier.pkl"


class Emojifier:
    def __init__(self, pred, W, b) -> None:
        self.pred = pred
        self.W = W
        self.b = b


def sentence_to_avg(sentence, word_to_vec_map):

    any_word = list(word_to_vec_map.keys())[0]

    words = sentence.lower().split()
    avg = np.zeros(word_to_vec_map[any_word].shape)

    count = 0
    for w in words:
        if w in list(word_to_vec_map.keys()):
            avg += word_to_vec_map[w]
            count += 1

    if count > 0:
        avg = avg/count

    return avg


def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=200):

    any_word = list(word_to_vec_map.keys())[0]
    cost = 0

    m = Y.shape[0]
    n_y = len(np.unique(Y))

    n_h = word_to_vec_map[any_word].shape[0]

    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    Y_oh = convert_to_one_hot(Y, C=n_y)

    for t in range(num_iterations):
        for i in range(m):

            avg = sentence_to_avg(X[i], word_to_vec_map)
            z = np.add(np.dot(W, avg), b)
            a = softmax(z)

            cost = -np.sum(np.dot(Y_oh[i], np.log(a)))

            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 10 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b


def predict_single(sentence, W, b, word_to_vec_map):

    any_word = list(word_to_vec_map.keys())[0]
    n_h = word_to_vec_map[any_word].shape[0]
    words = sentence.lower().split()
    avg = np.zeros((n_h,))
    count = 0
    for w in words:
        if w in word_to_vec_map:
            avg += word_to_vec_map[w]
            count += 1

    if count > 0:
        avg = avg / count

    Z = np.dot(W, avg) + b
    A = softmax(Z)
    pred = np.argmax(A)

    return pred


def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
