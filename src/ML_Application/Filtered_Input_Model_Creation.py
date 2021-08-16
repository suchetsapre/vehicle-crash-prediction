import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential


def positive_prediction(single_pred, thresh):
    ind = np.argmax(single_pred)
    return single_pred[ind] >= thresh


def collective_accuracy(y_test, pred, thresh):
    tp, tn, fp, fn = [0 for _ in range(4)]
    total = y_test.shape[0]

    for i in range(y_test.shape[0]):
        if 1 in y_test[i].tolist():
            if positive_prediction(pred[i], thresh):
                tp += 1
            else:
                fn += 1
        else:
            if positive_prediction(pred[i], thresh) is False:
                tn += 1
            else:
                fp += 1

    print('TP: %i, TN: %i, FP: %i, FN: %i' % (tp, tn, fp, fn))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * (precision * recall) / (precision + recall)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1-Score: %f' % f1score)


def create_and_train_ml_model():
    x_train, x_test, y_train, y_test = np.load('X_train_100pct.npy', allow_pickle=True), np.load('X_test_100pct.npy', allow_pickle=True), np.load('y_train_100pct.npy', allow_pickle=True), np.load('y_test_100pct.npy', allow_pickle=True)

    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            if type(x_train[i][j][0]) is np.ndarray:
                new_arr = []
                for k in range(x_train[i][j][0].shape[0]):
                    new_arr.append(x_train[i][j][0][k])
                while len(new_arr) < x_train.shape[2]:
                    new_arr.append(0)
                x_train[i][j] = np.array(new_arr)

    for i in range(x_test.shape[0]):
        for j in range(x_test.shape[1]):
            if type(x_test[i][j][0]) is np.ndarray:
                new_arr = []
                for k in range(x_test[i][j][0].shape[0]):
                    new_arr.append(x_test[i][j][0][k])
                while len(new_arr) < x_test.shape[2]:
                    new_arr.append(0)
                x_test[i][j] = np.array(new_arr)

    print(x_train[20])

    model = Sequential((
        Conv1D(filters=8, kernel_size=6, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
        Conv1D(filters=4, kernel_size=3, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(150, activation='relu'),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(y_train.shape[1], activation='sigmoid'),
    ))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    # model_save_file = 'SecondCondensedModelTest_LossOptimization.h5'
    model_save_file = 'test_model_for_output_graph.h5'
    patience = EarlyStopping(monitor='val_loss', patience=500)
    checkpoint = ModelCheckpoint(model_save_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # model.fit(X_train, y_train, epochs=1, batch_size=2, verbose=2, callbacks=[checkpoint, patience], validation_data=(X_test, y_test))

    model = keras.models.load_model(model_save_file)
    pred = model.predict(x_test)

    collective_accuracy(y_test, pred, 0.5)

    for i in range(x_test.shape[0]):
        if i % 10 != 0:
            continue
        plt.plot(pred[i], 'r', label='Predicted')
        plt.plot(y_test[i], 'b', label='True')
        plt.xlabel('Frame Number')
        plt.ylabel('Probability of Crash Within Next 20 Frames')
        plt.title('Filtered Data ML Model Graphical Output')
        plt.legend()
        plt.show()
