import numpy as np
import string
from keras.preprocessing.text import Tokenizer
# Teh whole idea of the exercise was to show character tokenization and string
# which can be possible by self or by keras utility function
# which would take care of many other factors as well

# Creating a on hot encoding for strings
samples = ['The cat sat on the mat.',
           'The dog ate my homework.']
token_index = {}
# quite simply take all the strings that present in teh sample set ad index them.
# later reverse index it with the index in the sample space
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length = 10
results = np.zeros(shape=(len(samples),
                        max_length,
                        max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1

print(token_index)
print(results)

# Creating a on hot encoding for characters
characters = string.printable
token_index = dict(zip(range(1, len(characters) + 1), characters))
# Here sip create a pair wise dict for the list generated from the range ouput
max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.

print(token_index)
print(results)

# Using Keras tokenizer method

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

print(one_hot_results)