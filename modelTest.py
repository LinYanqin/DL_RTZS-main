from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
# import keras
import h5py
import torch
# from modelTrain import ACResNet
# from keras.callbacks import ReduceLROnPlateau

# Read experimental data
# a = h5py.File('exp_quinine_1.mat','r')
# a = h5py.File('exp_quinine_2.mat')
a = h5py.File('exp_azithromycin.mat', mode='r')
# a = h5py.File('exp_strychnine.mat',mode='r')

# Pre-process
X = a['data']
num_raws = 4096
X = np.array(X).reshape(1, num_raws).astype('float16')
X = np.expand_dims(X, axis=1)
print('Data shape: ', X.shape)

# Do not need loss function when testing
def HADAMARD_YZX(y_true, y_pred):
    return 0

# Load model
model = load_model('model_rtzs.h5', custom_objects={'HADAMARD_YZX': HADAMARD_YZX})
# model = torch.load('model_rtzs.h5')
# print(model)

# model = ACResNet(4096, data_dims=1, num_output_channels=1)
# model.compile(loss=HADAMARD_YZX, optimizer='adam', metrics=['accuracy'])
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, mode='auto')
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00011,patience=4,verbose=0,mode='auto')
# callback = reduce_lr
# callback.set_model(model)
# model.load_weights('model_rtzs.h5', by_name=True)


# Post-process
predict = model.predict(X, verbose=1)

# Show
xt = X[0]
plt.subplot(211)
plt.plot(xt)
plt.subplot(212)
plt.plot(predict.reshape(num_raws, 1))
plt.show()