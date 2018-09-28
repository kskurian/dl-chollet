from keras.datasets import boston_housing
from keras import models
from keras import layers
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

#Feature normalization - where we use the trained data mean and standard deviation
mean = train_data.mean()
std = train_data.std()
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

