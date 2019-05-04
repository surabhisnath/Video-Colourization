from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import os
import cv2
# define input sequence

dim = (64, 36)
frames = os.listdir("../original/")
frames.sort()
imgs = []
for i in frames:
	# print(i)
	img = cv2.imread("../original/" + i)
	img = cv2.resize(img, dim)
	img = np.reshape(img, (img.shape[0]*img.shape[1]*img.shape[2]))
	# print(img.shape)
	imgs.append(img/255)

imgs = np.array(imgs)

# seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# print(imgs[0].shape)
seq_in = imgs[0]
# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# prepare output sequence
# seq_out = seq_in[:, 1:, :]
seq_out = imgs[1]

# n_out = n_in - 1
n_out = n_in
seq_out = seq_out.reshape((1, n_out, 1))
# define model
model = Sequential()
model.add(LSTM(6912, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_out))
model.add(LSTM(6912
	, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
#plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')
# fit model
print(seq_in.shape)
print(seq_out.shape)
print("Fitting Model ------------------------------")
model.fit(seq_in, seq_out, epochs=100, verbose=0)
# serialize model to JSON

model_json = model.to_json()
with open("model_100.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_100.h5")
print("Saved model to disk")

# demonstrate prediction
print("Predicting ---------------------------------")
yhat = model.predict(seq_in, verbose=0)
print(yhat[0,:,0])
print(seq_out[0,:,0])
print(yhat[0,:,0] - seq_out[0,:,0])
print("Saving -------------------------------------")
pred_img = np.resize(yhat[0,:,0]*255, (36, 64, 3))
cv2.imwrite("save_100.jpg", pred_img)
actual_img = np.resize(seq_out[0,:,0]*255, (36, 64, 3))
cv2.imwrite("actual.jpg", actual_img)
#-------------------
#Load model
'''
# load json and create model
json_file = open('model_100.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_100.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss='mse')
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''
