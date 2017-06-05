import cv2                
import matplotlib.pyplot as plt                        
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os.path import split
from glob import glob

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

files = glob("dogImages/test/*/*")

for path in files:
    print(path)
    d, f = split(path)
    prefix = f.split('.')[0]
    print(prefix)
    img = load_img(path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=d, save_prefix=prefix, save_format='jpg'):
        i += 1
        if i > 2:
            break

# gen_files = glob('preview/*')
# fig = plt.figure()
# for i, f in enumerate(gen_files):
#     a = fig.add_subplot(1, len(gen_files), i+1)
#     img = cv2.imread(f)
#     plt.imshow(img)
#     plt.axis('off')
# plt.show()
