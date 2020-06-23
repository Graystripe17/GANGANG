import numpy as np
from numpy.random import rand
from numpy.random import randint
from keras.datasets.mnist import load_data
from matplotlib import pyplot
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, LeakyReLU, Dropout, Flatten
from keras.utils.vis_utils import plot_model

def load_real_samples():
    (trainX, trainy), (testX, testy) = load_data()
    X = np.expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = X / 255.0
    return X

def generate_real_samples(dataset, n_samples):
    print(dataset.shape)
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y

def generate_fake_samples(n_samples):
    X = rand(28 * 28 * n_samples)
    X = X.reshape((n_samples, 28, 28, 1))
    y = np.zeros((n_samples, 1))
    return X, y

def define_discriminator(in_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train_discriminator(model, dataset, n_iter=100, n_batch=256):
    half_batch = int(n_batch / 2)
    for i in range(n_iter):
        X_real, y_real = generate_real_samples(dataset, half_batch)
        _, real_acc = model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(half_batch)
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
        print(">%d real=%.0f%% fake=%.0f%%" % (i+1, real_acc*100, fake_acc*100))


X = load_real_samples()


model = define_discriminator()
model.summary()
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
dataset = load_real_samples()
train_discriminator(model, dataset)
