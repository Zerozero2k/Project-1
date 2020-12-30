from keras.callbacks import Callback
from keras.models import load_model
from keras import backend as K
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Nhap i voi gia tri tu 0 den 1491 de chon thu muc
i = 2 

val_dir = '/home/thinhluong/Desktop/VideoPredict/catz/test'

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
model = load_model('ConvLSTM2D.h5', custom_objects={"perceptual_distance": perceptual_distance})

cat_dirs = glob.glob(val_dir + "/*")
input_images = np.zeros(
            (1, 96, 96, 3 * 5))
output_images = np.zeros((1, 96, 96, 3))
input_imgs = glob.glob(cat_dirs[i] + "/cat_[0-4]*")
imgs = [np.array(Image.open(img)) for img in sorted(input_imgs)]
input_images[0] = np.concatenate(imgs, axis=2)
output_img = glob.glob(cat_dirs[i] + "/cat_result.jpg")
output_images[0] = np.array(Image.open(output_img[0]))
input_images[0] /= 255.
output_images[0] /= 255.

input_image_seq = np.concatenate(imgs, axis=1)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(input_image_seq)
ax.set_title('5 Input Images')
plt.show()

image_6 = model.predict(input_images)[0]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(image_6)
ax.set_title('Image 6 prediction')
plt.show()

imgs.append(image_6)
imgs = imgs[1:6]
input_images[0] = np.concatenate(imgs, axis=2)

image_7 = model.predict(input_images)[0]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(image_7)
ax.set_title('Image 7 Prediction')
plt.show()

imgs.append(image_7)
imgs = imgs[1:6]
input_images[0] = np.concatenate(imgs, axis=2)

image_8 = model.predict(input_images)[0]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(image_8)
ax.set_title('Image 8 Prediction')
plt.show()
