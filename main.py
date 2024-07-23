import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
from modelutils import *



X_train, Y_train = read_csv(
    '.\\data\\train_emoji.csv')
X_test, Y_test = read_csv(
    '.\data\\tesss.csv')

maxLen = len(max(X_train, key=len).split())

# for idx in range(10):
#     print(X_train[idx], label_to_emoji(Y_train[idx]))

Y_oh_train = convert_to_one_hot(Y_train, C=5)
Y_oh_test = convert_to_one_hot(Y_test, C=5)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(
    ".\\data\\glove.6B.50d.txt")

pred, W, b = model(X_train, Y_train, word_to_vec_map)
emojifier = Emojifier(pred, W, b)
save_object(emojifier, file_name)

