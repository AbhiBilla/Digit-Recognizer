#Importing Dataset
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

# Normalization
x_train = x_train/255
x_test = x_test/255

# NN Architecture'
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = x_train[0].shape))
model.add(tf.keras.layers.Dense(784,activation="relu"))
model.add(tf.keras.layers.Dense(1568,activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# Compilation
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

#Model Fitting 
model.fit(x_train, y_train,epochs=10)

#Prediction
import numpy as np
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_pred

y_test

# Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_pred,y_test)
acc

#Model Saving
model.save("Minor_Project2.hdf5")