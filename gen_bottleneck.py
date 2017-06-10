
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import numpy as np
import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)

data_types = ['train', 'test', 'valid']
predictions = []

model = ResNet50(include_top=False)

for dt in data_types:
    img_paths = glob.glob("dogImages/" + dt + "/*/*")
    # img_paths = glob.glob("dogImages_copy/" + dt + "/*/*")
    # img_paths = glob.glob("dogImages_copy/" + dt + "/001.Affenpinscher/*")

    img_input = preprocess_input(paths_to_tensor(img_paths))

    print(img_input.shape)

    res = model.predict(img_input)
    predictions.append(res)

np.savez('test.npz', train=predictions[0], test=predictions[1], valid=predictions[2])
