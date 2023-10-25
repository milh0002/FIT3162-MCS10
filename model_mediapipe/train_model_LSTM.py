from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import datetime


def generate_model(actions):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30, 132)))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.summary()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def fit_data(model, X_train, y_train):
    early_stopping_monitor = EarlyStopping(patience=40)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2000,
              callbacks=[tensorboard_callback, early_stopping_monitor], batch_size=64)


def test_model(model, X_test):
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    print(multilabel_confusion_matrix(ytrue, yhat))

if __name__ == "__main__":
    actions = np.array(["Sitting", "Standing", "Walking"])

    X = np.load("X1.npy")
    y = np.load("y1.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = generate_model(actions)
    fit_data(model, X_train, y_train)
    test_model(model, X_test)


    model.save("action1_1.h5")
