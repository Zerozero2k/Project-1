from keras.callbacks import Callback
from keras.models import load_model
from keras import backend as K
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

val_dir = '/home/thinhluong/Desktop/Project_1/test/0001'

# Function for measuring how similar two images are

def perceptual_distance(y_true, y_pred):
    y_true *= 255.
    y_pred *= 255.
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

# load model
model = load_model('ConvLSTM2DModelAtari', custom_objects={"perceptual_distance": perceptual_distance})

i = 2

cat_dirs = glob.glob(val_dir + "/*")
input_images = np.zeros(
            (1, 210, 160, 3 * 5))
output_images = np.zeros((1, 210, 160, 3))
input_imgs = glob.glob(cat_dirs[i] + "/cat_[0-4]*")
imgs = [np.array(Image.open(img)) for img in sorted(input_imgs)]
input_images[0] = np.concatenate(imgs, axis=2)
output_img = glob.glob(cat_dirs[i] + "/cat_5*")
output_images[0] = np.array(Image.open(output_img[0]))
input_images[0] /= 255.
output_images[0] /= 255.

input_image_seq = np.concatenate(imgs, axis=1)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(input_image_seq)
ax.set_title('5 Input Images')
plt.show()

image_6 = np.array(model.predict(input_images)[0])
image_tmp = glob.glob(cat_dirs[i] + "/cat_5*")
image_output_6 = np.array(Image.open(image_tmp[0]))
fig, ax= plt.subplots(1, 2)
ax[0].imshow(image_6)
ax[0].set_title('Image 6 prediction')
ax[1].imshow(image_output_6)
ax[1].set_title('Image 6 output')
plt.show()

imgs.append(image_6)
imgs = imgs[1:6]
input_images[0] = np.concatenate(imgs, axis=2)

image_7 = np.array(model.predict(input_images)[0])
image_tmp = glob.glob(cat_dirs[i] + "/cat_6*")
image_output_7 = np.array(Image.open(image_tmp[0]))
fig, ax= plt.subplots(1, 2)
ax[0].imshow(image_7)
ax[0].set_title('Image 7 prediction')
ax[1].imshow(image_output_7)
ax[1].set_title('Image 7 output')
plt.show()

imgs.append(image_7)
imgs = imgs[1:6]
input_images[0] = np.concatenate(imgs, axis=2)

image_8 = np.array(model.predict(input_images)[0])
image_tmp = glob.glob(cat_dirs[i] + "/cat_7*")
image_output_8 = np.array(Image.open(image_tmp[0]))
fig, ax= plt.subplots(1, 2)
ax[0].imshow(image_8)
ax[0].set_title('Image 8 prediction')
ax[1].imshow(image_output_8)
ax[1].set_title('Image 8 output')
plt.show()
