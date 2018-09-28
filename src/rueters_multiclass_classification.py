from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(len(train_data))
print(len(test_data))

print(train_data[10])

word_indexs = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_indexs.items()])
decode_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[10]])

print(decode_newswire)
print(train_labels[10])


def vectorize_sequences( sequence, dimension=10000):
    results = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequence):
        results[i, sequence] = 1
    return results

print("vectorizing train")
x_train = vectorize_sequences(train_data)
print(len(x_train))
print(x_train.ndim)

print("vectorizing test")
x_test = vectorize_sequences(test_data)
print(len(x_test))
print(x_test.ndim)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


print("One hot encoding")
one_hot_train_labels = to_one_hot(train_labels)
print("Train label 10 ")
print(one_hot_train_labels[10])
one_hot_test_labels = to_one_hot(test_labels)
print("Test label 10 ")
print(one_hot_test_labels[10])


#There is a inbuilt way to do the same in keras
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

print("Model Complilation is over")

# print("x_train")
# print(len(x_train))
x_val = x_train[:1000]
# print("x_val")
# print(len(x_val))
partial_x_train = x_train[1000:]
# print("x_trian")
# print(len(partial_x_train))

# print("y_trian")
# print(len(one_hot_train_labels))
y_val = one_hot_train_labels[:1000]
# print("y_val")
# print(len(y_val))
partial_y_train = one_hot_train_labels[1000:]
# print("partial_y_train")
# print(len(partial_y_train))

history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs,val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()