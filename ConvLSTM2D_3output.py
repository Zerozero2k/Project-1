from keras.callbacks import Callback
from keras import backend as K

import random
import glob
import wandb
from wandb.keras import WandbCallback
from PIL import Image
import numpy as np

#initialize wandb

hyperparams = {"num_epochs": 10,
          "batch_size": 32,
          "height": 210,
          "width": 160}

wandb.init(config=hyperparams)
config = wandb.config

val_dir = '/home/thinhluong/Desktop/Project_1/test/0001'
train_dir = '/home/thinhluong/Desktop/Project_1/train/0001'

# generator to loop over train and test images

def my_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, config.height, config.width, 3 * 5))
        output_images = np.zeros((batch_size, config.height, config.width, 3))
        random.shuffle(cat_dirs)
        if (counter+batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-4]*")       
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            input_images[i] = np.concatenate(imgs, axis=2)
            output_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[5-7]*")
            imgs = [Image.open(img) for img in sorted(output_imgs)]
            output_images[i] = np.concatenate(imgs, axis=1)
            input_images[i] /= 255.
            output_images[i] /= 255.
        yield (input_images, output_images)
        counter += batch_size

#callback to log the images

class ImageCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        validation_X, validation_y = next(
            my_generator(15, val_dir))
        output = self.model.predict(validation_X)
        wandb.log({
            "input": [wandb.Image(np.concatenate(np.split(c, 5, axis=2), axis=1)) for c in validation_X],
            "output": [wandb.Image(np.concatenate([validation_y[i], o], axis=1)) for i, o in enumerate(output)]
        }, commit=False)


# Function for measuring how similar two images are

def perceptual_distance(y_true, y_pred):
    y_true *= 255.
    y_pred *= 255.
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

# ConvLSTM2D predict 3 next frame

from keras.layers import Lambda, Reshape, Permute, Input, add, Conv2D, TimeDistributed, Conv2DTranspose, MaxPooling2D, ConvLSTM2D, Concatenate
from keras.models import Model

inp          = Input((config.height, config.width, 5 * 3))
reshaped     = Reshape((config.height,config.width, 5, 3))(inp)
permuted     = Permute((1,2,4,3))(reshaped)
permuted_2   = Permute((4,1,2,3))(permuted)


convLSTM2D_1   = ConvLSTM2D(3, (3,3), padding="same", return_sequences=True)(permuted_2)    # Cho ra anh input 1-5 va anh 6 du doan 
convLSTM2D_1   = Lambda(lambda x : x[:,1:,:,:,:])(convLSTM2D_1)                             # Bo di anh input 1 ra khoi dau vao

convLSTM2D_2   = ConvLSTM2D(3, (3,3), padding="same", return_sequences=True)(convLSTM2D_1)  # Cho ra anh input 2-5 va anh 6 du doan va anh 7 du doan
convLSTM2D_2   = Lambda(lambda x : x[:,1:,:,:,:])(convLSTM2D_2)                             # Bo di anh input 2 ra khoi dau vao

convLSTM2D_3   = ConvLSTM2D(3, (3,3), padding="same", return_sequences=True)(convLSTM2D_2)  # Cho ra anh input 3-5 va anh 6-8 du doan
convLSTM2D_3   = Lambda(lambda x : x[:,3:,:,:,:])(convLSTM2D_3)                             # Lay ra anh 6,7,8 du doan

mergered       = Concatenate(axis=1)([ convLSTM2D_3[:,0,:,:,:], convLSTM2D_3[:,1,:,:,:], convLSTM2D_3[:,2,:,:,:] ])     # Y dinh chap 3 anh cuoi theo chieu ngang


model=Model(inputs=[inp], outputs=[mergered])

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config.batch_size, train_dir),
                    steps_per_epoch=len(glob.glob(train_dir + "/*")) // config.batch_size,
                    epochs=config.num_epochs, callbacks=[
    ImageCallback(), WandbCallback()],
    validation_steps=len(glob.glob(val_dir + "/*")) // config.batch_size,
    validation_data=my_generator(config.batch_size, val_dir))