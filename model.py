
# coding: utf-8

# In[201]:

import numpy as np
import pandas as pd
import random
import math
import csv
import cv2
import os,sys
import json
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

#import scikit-learn functions
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[12]:

print(os.getcwd())
#get_ipython().system('ls -lh')


# In[428]:

""" 
CONSTANTS 
"""

root_dir = os.getcwd()
path2data = root_dir + '/data'
csv_driving_data = path2data+'/driving_log.csv'

# Data augmentation constants
TRANSLATION_ANGLE = 0.3 #Max angle change when translating in the X direction
TRANSLATION_X_DIR = 100 #Number of translation pixels in the X direction
TRANSLATION_Y_DIR =  40 #Number of translation pixesl in the Y direction
ANGLE_CORRECTION_LR_CAMERA = 0.25 #Angle correction for left/right camera

BRIGHTNESS_CORRECTION = 0.25 #Brightness correction applied to an image during preprocess
ANGLE_LIMIT = 1.0 #Absolute value of steering angle cannot be more than this
ANGLE_BIAS = 0.5

#Training Constants
BATCH_SIZE = 256  #Number of images in a batch
EPOCH = 10  #number of epochs to train the model
TRAIN_BATCH_PER_EPOCH = 160 #number of batches of samples in each epoch
CURRENT_EPOCH_NUMBER = 0 # keep track of the epoch number, initially 0, after EPOCH, 9

#Image related Constants
IMG_ROWS = 66
IMG_COLS = 200
IMG_CHNL = 3




df = pd.read_csv(csv_driving_data)
#make a copy of the df just in case I need it later
df_raw = df.copy() #makes a deep copy

# In[362]:

print(df.shape)

#split the data into two, for resampling purpose
DF_zero = df[(df.steering == 0)]
DF_nonzero = df[(df.steering != 0)]

"""
sample from DF_zero a smaller subset 
since the number of 0s have a significantly higher representation, the model prediction
is heavily biased towards 0 and (possibly going off the road while turning)
so instead of randomly skipping an image while preprocessing, I changed my mind, and resampling
here so to make sure that the we have a more even histogram
"""
DF_zero = DF_zero.sample(np.int(DF_zero.shape[0]*(1.0/9)))
concatDF = pd.concat([DF_zero,DF_nonzero])


"""instead of randomly selecting a camera position (as in camera_choice function), I will use all the
available data, merging the camera positions (center,left,right) into a single column 'camera'
and suitably adjusting the steering value for the left/right camera positions, This will be starting
for the data set
"""
df_center = concatDF[[0,3,4,5,6]]
df_left = concatDF[[1,3,4,5,6]]
df_right = concatDF[[2,3,4,5,6]]

#rename the columns so that vertically combining the datasets is possible
df_center.columns = ['camera','steering','throttle','brake','speed']
df_left.columns = ['camera','steering','throttle','brake','speed']
df_right.columns = ['camera','steering','throttle','brake','speed']

""" 
Now fix the steering values for the left and right camera positions to shift the positions to the 
center
"""

df_left.iloc[:,0] = df_left.steering.apply(lambda x: x + ANGLE_CORRECTION_LR_CAMERA)
df_right.iloc[:,0] = df_right.steering.apply(lambda x: x - ANGLE_CORRECTION_LR_CAMERA)

#Now marge, reindex

df = pd.concat([df_center,df_left,df_right])
df.index = range(len(df))

# In[236]:

# plt.subplot(2,1,1)
# plt.plot(df['steering'])
# plt.title('steering values plotted over time')
# plt.subplots_adjust(hspace = 0.6)
# plt.subplot(2,1,2)
# plt.plot(df['steering'].iloc[3400:4300])
# plt.title('steering values between 3400 and 4300')


# In[64]:

# g = plt.hist(df['steering'], 100)


# In[363]:

#randomly chooses a camera angle 
#outputs: the path to the camera in the file folder and shift_angle that we add/subtract from
#         angle to correct the direction of driving/steering angle
# def camera_choice(df):
#     camera=random.choice(['left','center','right'])
#     if camera_choice == 'left':
#         img_path = os.path.join(path2data,df[camera].iloc[0].strip())
#         shift_angle = ANGLE_CORRECTION_LR_CAMERA
#     elif camera_choice == 'center':
#         img_path = os.path.join(path2data, df[camera].iloc[0].strip())
#         shift_angle = 0
#     else:
#         img_path = os.path.join(path2data, df[camera].iloc[0].strip())
#         shift_angle = -1*ANGLE_CORRECTION_LR_CAMERA
#     return img_path, shift_angle, camera 


# In[256]:
"""image_path = os.path.join(path2data,df["center"].iloc[0].strip())
image = cv2.imread(image_path)
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image1 = np.float32(image1)
random_brightness = .25 + np.random.uniform()
image1[:,:,2] *= random_brightness
image1 = np.uint8(image1)"""


# image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

# plt.imshow(image1)



# In[295]:

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.float32(image1)
    random_brightness = BRIGHTNESS_CORRECTION + np.random.uniform()
    image1[:,:,2] *= random_brightness
    image1 = np.uint8(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1 
#cv2.cvtColor(np.fliplr(img), cv2.COLOR_BGR2RGB)

# img = augment_brightness_camera_images(image1)
# plt.imshow(np.hstack((image1,cv2.cvtColor(img,cv2.COLOR_BGR2RGB))))


# In[258]:

#utility function to visualize augmented figures
#usage: visualize_aug_image(image,FUN=augment_brightness_camera_images)
def visualize_aug_image(image,FUN):
    fig, axs = plt.subplots(4,5, figsize = (10,4))
    fig.subplots_adjust(hspace = 0.5,wspace = 0.2)
    axs = axs.ravel()
    for i in range(20):
        axs[i].axis('off')
        img1 = FUN(image)
        #axs[i].imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
        axs[i].imshow(img1)
    #axs[i].set_title(df.steering.iloc[rand_idx])


# In[259]:

#visualize augmented images of a randomly selected image
"""rand_idx = random.randint(0,len(df)) 
camera = random.choice(['left','center','right'])
img_path = os.path.join(path2data, df[camera].iloc[rand_idx].strip())
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
visualize_aug_image(image,FUN=augment_brightness_camera_images)
"""

# In[261]:

#flip image and the corresponding steering angle 
def flip_image_random(image,new_angle):
    image = np.fliplr(image)
    new_angle = -new_angle
    return image, new_angle
        


# In[262]:
"""rand_idx = random.randint(0,len(df)) 
camera = random.choice(['left','center','right'])
img_path = os.path.join(path2data, df[camera].iloc[rand_idx].strip())
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img, angle = flip_image_random(image,df['steering'].iloc[rand_idx])
print(angle)
plt.imshow(np.hstack((image,img)))
plt.title('angle after flipping:{}'.format(angle))

"""
# In[291]:

#horizontal and vertical shift 
def image_translation(image, angle, translation_range=TRANSLATION_X_DIR):
    x_translation = (translation_range * np.random.uniform()) - (translation_range/2)
    new_angle = angle + (x_translation/translation_range)*2*TRANSLATION_ANGLE
    y_translation = TRANSLATION_Y_DIR*np.random.uniform() - TRANSLATION_Y_DIR/2
    Trans_Mat = np.float32([[1,0,x_translation],[0,1,y_translation]])
    image_translation = cv2.warpAffine(image,Trans_Mat, (image.shape[1],image.shape[0]))
    return image_translation, new_angle, x_translation


# In[264]:

"""idx = np.random.randint(len(df))
img_path = os.path.join(path2data, df.left.iloc[idx].strip())
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
steer_angle = df.steering.iloc[idx]

fig, axs = plt.subplots(4,5, figsize = (12,6))
fig.subplots_adjust(hspace =0.5, wspace = 0.2)
axs = axs.ravel()
for i in range(20):
    axs[i].axis('off')
    img1,steer_angle,trans_x = image_translation(image, steer_angle,80)
    axs[i].imshow(img1)
    axs[i].set_title('steer:{} trx={}'.format(str(np.round(steer_angle,2)),str(np.round(trans_x,2))))
"""

# In[265]:

#crop and resize
#new_col_length = #IMG_COLS
#new_row_length = #IMG_ROWS

def crop_and_resize(image,add_blur = True):
    shape = image.shape
    #crop the image to remove unncessary top and bottom parts 
    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    if add_blur:
        #add some blur 
        image = cv2.GaussianBlur(image, (3,3),0)
    #resize to NVIDIA requirements
    image = cv2.resize(image, (IMG_COLS, IMG_ROWS), interpolation=cv2.INTER_AREA)
    return image

"""img_copy = image #image from above
plt.subplot(2,1,1)
img = crop_and_resize(image)
plt.imshow(img)
plt.axis('off')
plt.title('cropped image:{}x{}'.format(img.shape[0],img.shape[1]))
plt.subplot(2,1,2)
plt.imshow(img_copy)
plt.axis('off')
plt.title('original image:{}x{}'.format(img_copy.shape[0],img_copy.shape[1]))"""


# In[266]:

"""idx_seq = [np.random.randint(0,len(df)) for i in range(20)]
fig,axs = plt.subplots(4,5, figsize=(12,10))
fig.subplots_adjust(hspace =0.2, wspace = 0.1)
axs = axs.ravel()
for i,idx in zip(range(20),idx_seq):
    axs[i].axis('off')
    img_path = os.path.join(path2data,df.left.iloc[idx].strip())
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_copy = crop_and_resize(image)
    axs[i].imshow(img_copy)
    #axs[i].set_title('steer:{},index:{}'.format(str(df.steering.iloc[idx]), str(idx)))
"""

# In[379]:

#bring the augmentation functions together to preprocess

def preprocess_image(data, bias = ANGLE_BIAS, testing_flag = False):    
    if not testing_flag:
        #path_file, shift_angle,camera = camera_choice(data)
        path_file, y_steer = os.path.join(path2data,data.camera[0].strip()), data.steering[0]
        image = cv2.imread(path_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # y_steer = data['steering'].iloc[0]+shift_angle
        # threshold = np.random.uniform()
        # if abs(y_steer)+bias < threshold:
        #     return None,None,None
        image, y_steer, tr_x = image_translation(image, y_steer, 150)
        image = augment_brightness_camera_images(image)
        image = crop_and_resize(image)
        image = np.array(image)
        #image, y_steer = flip_image_random(image,y_steer)
        return image, y_steer,camera
    else:
        path_file = os.path.join(path2data, data['camera'].iloc[0].strip())
        image = cv2.imread(path_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_and_resize(image)
        y_steer = data['steering'].iloc[0]
        return image, y_steer


# In[364]:

"""data = df.iloc[[2]].reset_index()
plt.figure(figsize=(16,8))
ii = 0
while ii < 32:
    image, steer,camera = preprocess_image(data)
    if image is not None:
        plt.subplot(4,8,ii+1);
        plt.axis('off')
        plt.imshow(image)
        plt.title('{}:{}'.format(str(np.round(steer,2)),camera))
        ii += 1
        """


# In[421]:

#generate training data
def generate_data_batch(df, training_flag = True):
    global CURRENT_EPOCH_NUMBER
    x = np.zeros([BATCH_SIZE,IMG_ROWS,IMG_COLS,IMG_CHNL],dtype=np.float)
    y = np.zeros([BATCH_SIZE], dtype=np.float)
    
    #num_batches = 1
    count = 0
    while True: 
        idx = np.random.randint(len(df))
        data = df.iloc[[idx]].reset_index()
        if training_flag:
            image,steer_angle,_ = preprocess_image(data, bias = 1.0/(1.0+CURRENT_EPOCH_NUMBER))

            
            if np.random.randint(2)==1:
                image, steer_angle = flip_image_random(image,steer_angle)
            x[count] = image
            y[count] = steer_angle
            count += 1
            if count >= BATCH_SIZE:
                # if num_batches >= TRAIN_BATCH_PER_EPOCH:
                #     CURRENT_EPOCH_NUMBER += 1
                #     num_batches = 1
                # else:
                #     num_batches +=1
                yield (x,y)
                x = np.zeros([BATCH_SIZE,IMG_ROWS,IMG_COLS,IMG_CHNL],dtype=np.float)
                y = np.zeros([BATCH_SIZE], dtype=np.float)
                count = 0
        else:
            image,steer_angle = preprocess_image(data,testing_flag=True)
            x[count] = image
            y[count] = steer_angle
            count += 1
            if count >= BATCH_SIZE:
                yield (x,y)
                x = np.zeros([BATCH_SIZE,IMG_ROWS,IMG_COLS,IMG_CHNL],dtype=np.float)
                y = np.zeros([BATCH_SIZE], dtype=np.float)
                count = 0
    


            
    


# In[414]:

#function to shuffle data frame before splitting
def df_shuffle(df):
    #DF = df.reindex(np.random.permutation(df.index))
    n = len(df)
    DF = df.sample(n)
    DF.index = range(n)
    return DF


# In[415]:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Input, ELU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras import initializations
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

import tensorflow as tf
tf.python.control_flow_ops = tf


# In[416]:

#Model parameters
input_shape = (IMG_ROWS,IMG_COLS,IMG_CHNL)
#filter_size = 3
#pool_size = (2,2)


# In[425]:

def build_model():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.0 - 1.0, input_shape = input_shape))

    #COLOR MAP LAYER 
    model.add(Convolution2D(3,1,1,border_mode = 'valid', name='conv0'))
    

    model.add(Convolution2D(24,3,3,activation='elu',border_mode = 'valid',subsample = (2,2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(36,3,3,activation='elu',border_mode = 'valid',subsample = (2,2)))
    model.add(Dropout(0.2))


    model.add(Convolution2D(48,3,3,activation='elu',border_mode = 'valid',subsample = (2,2)))
    model.add(Dropout(0.2))

    
    model.add(Convolution2D(64,3,3,activation='elu',border_mode = 'valid',subsample = (2,2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64,3,3,activation='elu',border_mode = 'valid'))
    model.add(Dropout(0.2))


    model.add(Flatten())

    model.add(Dense(100, activation = 'elu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, name = 'output'))
    
    model.summary()
    
    return model


# In[426]:

# def save_model(model):
#     with open('log/model.json','w') as f:
#         f.write(model.to_json())
#     model.save_weights('checkpoints/model.h5')
#     print("Model saved to /log")
        


# In[429]:

if __name__ == '__main__':
    model = build_model()

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay =0.0)
    model.compile(optimizer = adam, loss='mse')


    
        
    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:0.4f}.hdf5',
        save_best_only=True)
    logger = CSVLogger(filename='log/history.csv')
    Early_stopper = EarlyStopping(monitor='val_loss',min_delta=0)

    df = df_shuffle(df)
    train_data, val_data = train_test_split(df, test_size = 0.2, random_state=144)


    
    history = model.fit_generator(generator=generate_data_batch(train_data),
                       samples_per_epoch = TRAIN_BATCH_PER_EPOCH*BATCH_SIZE,
                       nb_epoch = 5,
                       validation_data = generate_data_batch(val_data,training_flag=False),
                       nb_val_samples= 100*BATCH_SIZE,
                       callbacks = [checkpointer,logger,Early_stopper],
                       verbose=1)


    with open('log/model.json','w') as f:
        f.write(model.to_json())

# In[ ]:



