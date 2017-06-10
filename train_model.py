
import numpy as np
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  

model_name = 'Resnet50'

# train_data = np.load('bottleneck_features_train.npy')
# valid_data = np.load('bottleneck_features_validation.npy')
train_data = np.load('aug_train.npy')
valid_data = np.load('aug_valid.npy')
# train_targets = np_utils.to_categorical(np.load('labels_train.npy'), 133)
# valid_targets = np_utils.to_categorical(np.load('labels_validation.npy'), 133)
train_targets = np.load('aug_train_labels.npy')
valid_targets = np.load('aug_valid_labels.npy')

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
model.add(Dense(133, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.' + model_name + '.hdf5', 
                                verbose=1, save_best_only=True)

model.fit(train_data, train_targets,
          validation_data=(valid_data, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
