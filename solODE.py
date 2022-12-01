#Paqueterias para el proyecto
import tensorflow as tf
import matplotlib
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

class SolODE(Sequential):
    loss_tracker = keras.metrics.Mean(name="loss")
class SolODE(Sequential):
    loss_tracker = keras.metrics.Mean(name="loss")

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size, 1), minval=-5, maxval=5)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self(x, training=True)
            dy = tape2.gradient(y_pred, x)
            x_0 = tf.zeros((batch_size, 1))
            y_0 = self(x_0, training=True)
            eq = x * dy + y_pred - x ** 2 * keras.backend.cos(x)
            ic = y_0
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

        @property
        def metrics(self):
            return [keras.metrics.Mean(name='loss')]

model = SolODE()
model.add(Dense(10, activation='tanh', input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])
tf.keras.layers.Dropout(.25, input_shape=(2,))
x=tf.linspace(-5,5,2000)
history=model.fit(x, epochs=2000, verbose=1)
x_testv = tf.linspace(-5, 5, 2000)

y = [((x*np.sin(x))+(2*np.cos(x))-((2/x)*np.sin(x))) for x in x_testv]

a = model.predict(x_testv)

plt.grid()
plt.title('Red vs Analitica')

plt.plot(x_testv, a)
plt.plot(x_testv, y)
plt.show()

model.save("red.h5")
exit()