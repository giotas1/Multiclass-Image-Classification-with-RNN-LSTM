# Step 1: Install Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Data Preprocessing

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test)= mnist.load_data() # loading the dataset


x_train.shape , x_test.shape
y_train.shape, y_test.shape
x_train.max()

# normalize the dataset
x_train= x_train/255.0
x_test=x_test/255.0

plt.imshow(x_train[5])


# Step 3: Building the LSTM

# Initialize the model
model = tf.keras.models.Sequential()

# Add layers to the model
model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(28, 28)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Print the summary of the model
model.summary()


opt= tf.keras.optimizers.Adam(learning_rate=0.001)

# compile the model
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Training the model
history =  model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# predictions
y_pred= model.predict(x_test)

print(y_pred[10]), print(y_test[10])


from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = np.argmax(y_pred_prob, axis=1)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(cm)
print(accuracy)


# Step 5: Learning Curve
def learning_curve(history, epoch):
    epoch_range= range(1, epoch + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'val'], loc= 'upper left')
    plt.show()

    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'val'], loc= 'upper left')
    plt.show()

learning_curve(history, 10)
