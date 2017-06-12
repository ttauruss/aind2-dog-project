
import numpy as np
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint  
from keras import applications
from keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

base_model = applications.ResNet50(include_top=False)
x = base_model.output
# x = GlobalAveragePooling2D(input_shape=(1,1,2048)))(x)
x = GlobalAveragePooling2D()(x)
x = Dense(133, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)
model.summary()

train_datagen = image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dogImages/train',
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical')

valid_generator = train_datagen.flow_from_directory(
    'dogImages/valid',
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical')

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50Aug.hdf5', verbose=1, save_best_only=True)

model.fit_generator(train_generator,
          validation_data=valid_generator,
	      steps_per_epoch=6680//32,
		  validation_steps=835//32,
          epochs=100, callbacks=[checkpointer], verbose=1)

