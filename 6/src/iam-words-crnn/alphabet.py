import numpy as np

def load_alphabet():
    alphabet = ""
    with open("dataset/alphabet.txt", "r") as file:
        alphabet = file.read()
    return alphabet.strip()

alphabet = load_alphabet()

def alphabet_inverse(alph):
    res = {}
    for (idx, ch) in enumerate(alph):
        res[ch] = idx
    return res

alphabet_inv = alphabet_inverse(alphabet)

def chars_to_ints(word):
    return [alphabet_inv[elem] for elem in word]

def ints_to_chars(nums):
    return "".join([alphabet[i] if i < len(alphabet) else "+" for i in nums])

def encode_texts(y):
    y_nums = np.ones([len(y), max([len(word) for word in y])]) * len(alphabet)
    for (idx, word) in enumerate(y):
        y_nums[idx][:len(word)] = chars_to_ints(word)

    return y_nums
