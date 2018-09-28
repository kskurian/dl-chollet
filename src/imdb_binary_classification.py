from keras.datasets import imdb
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# The argument num_words=10000 means you’ll only keep the top 10,000 most frequently occurring words

print(train_data[0])
# prints the review as a integer replaced from the list of words
print(train_labels[0])
print(" train label dimension : ")
print(train_labels.ndim)
print(train_labels.shape)

print(" train data dimensions :")
print(train_data.ndim)
print(train_data.shape)

#print(test_data[0])
#print(test_labels[0])

# you get the word index ie which word is indexed to a number
word_index = imdb.get_word_index()
# now each index would have a word associated
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 0, 1, and 2 are reserved indices for “padding,” “start of sequence,” and “unknown
decoded_review = ' '.join([reversed_word_index.get(i-3, '?') for i in train_data[0]])

print(decoded_review)


def vectorize_seq(sequence, dimension=10000):
    results = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequence):
        results[i, sequence] = 1
    return results


x_train = vectorize_seq(train_data)
x_test = vectorize_seq(test_data)

# if we see the initial index those would be 1 the reason being they are arranged in the latest order
print(x_train[0][100])
print(x_train[2][100])
print(x_train[9000][100])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(y_train[0])
print(y_train[2])
print(y_train[9000])

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#Here we are spliting the training set into 10000 and 15000 so that we can test the result with the remaining data
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

#Here we decide the optimizer - rmsprop , the loss function - binary crossentropy and the metric - accuracy
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# we define our data then the number of loops and the batch size for mini batch
history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))

# history_dict.keys() has [u'acc', u'loss', u'val_acc', u'val_loss']

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, 4 + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("The model evaluated on test data")

results = model.evaluate(x_test, y_test)
print("The results after evaluation on the test data is ")
print(results)

print("The predicting on x_test : ")
results = model.predict(x_test)

print("results are : ")
print(results)