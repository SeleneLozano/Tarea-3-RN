import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

loss_tracker = keras.metrics.Mean(name="loss")


class ODEsolver(Sequential):
     def train_step(self, data):
         # Unpack the data. Its structure depends on your model and
         # on what you pass to `fit()`.
         x = tf.random.uniform((80, 1), minval=-5, maxval=5)

         with tf.GradientTape() as tape:
             # Compute the loss value
             with tf.GradientTape() as tape2:
                 tape2.watch(x)
                 y_pred = self(x, training=True)
             dy = tape2.gradient(y_pred, x)
             x_o = tf.zeros((80, 1))
             y_o = self(x_o, training=True)
             eq = x*dy + y_pred - x**2*keras.backend.cos(x)
             ic = y_o
             loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)

        # Compute gradients
         trainable_vars = self.trainable_variables
         gradients = tape.gradient(loss, trainable_vars)
        # Update weights
         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        #self.compiled_metrics.update_state(y, y_pred)
         loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
         return {m.name: m.result() for m in self.metrics}

     @property
     def metrics(self):
       # We list our `Metric` objects here so that `reset_states()` can be
       # called automatically at the start of each epoch
       # or at the start of `evaluate()`.
       # If you don't implement this property, you have to call
       # `reset_states()` yourself at the time of your choosing.
       return [loss_tracker]


model = ODEsolver()

model.add(Dense(10, activation='tanh', input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])

x = tf.linspace(-5, 5, 100)
history = model.fit(x, epochs=500, verbose=1)

x_testv = tf.linspace(-5, 5, 100)
a = model.predict(x_testv)
plt.plot(x_testv, a)
plt.plot(x_testv, np.exp(-x*x))
plt.show()
exit()

#Para guardar el modelo en disco
model.save("red.h5")

#para cargar la red:
modelo_cargado = tf.keras.models.load_model('red.h5')
