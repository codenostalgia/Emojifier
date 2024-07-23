import pickle
from emo_utils import predict, label_to_emoji, read_glove_vecs
from modelutils import predict_single, save_object


class Emojifier:
    def __init__(self, word_to_vec_map, pred, W, b) -> None:
        self.pred = pred
        self.W = W
        self.b = b
        self.word_to_vec_map = word_to_vec_map


file_name = ".\emojifier.pkl"
file_name2 = ".\sentence_emojifier.pkl"
file = open(file_name2, "rb")


# word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(
#     "D:\\installations\\Coding\\Python\\Python course\\Emojifier\\data\\glove.6B.50d.txt")

# myemojifier = Emojifier(word_to_vec_map, emojifier.pred,
#                         emojifier.W, emojifier.b)

# save_object(myemojifier, file_name2)


emojifier = pickle.load(file, encoding='utf8')

sentence = input("Input: ")
print("Output: ", end="")
print(sentence+label_to_emoji(int(predict_single(sentence,
      emojifier.W, emojifier.b, emojifier.word_to_vec_map))))
