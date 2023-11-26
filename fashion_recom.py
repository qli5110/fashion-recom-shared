import numpy as np
import pandas as pd
import os 
import tensorflow as tf
import keras as keras
from keras import Model
from keras.applications.densenet import DenseNet121
from keras.applications import vgg16
from keras.applications import ResNet50
from keras.applications import ResNet152
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pathlib
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

#--------------------------------Data Processing----------------------------
print('tf version')
print(tf.__version__)
print('gpu available?')
tf.test.is_gpu_available()

path = './fashion-dataset/'
dataset_path = pathlib.Path(path)
dirs_names = os.listdir(dataset_path) # list content of dataset
dirs_names

styles_df = pd.read_csv(path + "styles.csv", nrows=6000, on_bad_lines='skip') # Read 6000 product and drop bad lines 
styles_df['image'] = styles_df.apply(lambda x: str(x['id']) + ".jpg", axis=1) # Make image column contains (id.jpg)
print(styles_df.shape)
print('styles_df content:\n',styles_df.head(5))

#--------------------------------Modeling and Get Embeddings-------------------------
img_width, img_height, chnls = 100, 100, 3

from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print('gpu info:', get_available_devices())


#Resnet50
resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, chnls))
resnet50.trainable=False
resnet50_model = keras.Sequential([resnet50, GlobalMaxPooling2D()])
resnet50_model.summary()


#Resnet152
resnet152 = ResNet152(include_top=False, weights='imagenet', input_shape=(img_width, img_height, chnls))
resnet152.trainable=False
resnet152_model = keras.Sequential([resnet152, GlobalMaxPooling2D()])
resnet152_model.summary()

MODEL = resnet50_model

def img_path(img):
    """ Take image name(id) and return the complete path of it """
    return path + 'images/' + img

def predict(model, img_name):
    """ Load and preprocess image then make prediction """
    # Reshape
    img = image.load_img(img_path(img_name), target_size=(img_width, img_height))
    # img to Array
    img = image.img_to_array(img)
    # Expand Dim (1, w, h)
    img = np.expand_dims(img, axis=0)
    # Pre process Input
    img = preprocess_input(img)
    return model.predict(img)

def get_embeddings(df, model):
    """ Return a dataframe contains images features """
    df_copy = df
    df_embeddings = df_copy['image'].apply(lambda x: predict(model, x).reshape(-1))
    df_embeddings = df_embeddings.apply(pd.Series)
    return df_embeddings

def get_similarity(model):
    """ Get similarity of custom image """
    sample_image = predict(MODEL, file_number)
    df_sample_image = pd.DataFrame(sample_image)
    sample_similarity = cosine_similarity(df_sample_image, df_embeddings)
    return sample_similarity

def normalize_sim(similarity):
    """ Normalize similarity results """
    x_min = similarity.min(axis=1)
    x_max = similarity.max(axis=1)
    norm = (similarity-x_min)/(x_max-x_min)[:, np.newaxis]
    return norm

def get_recommendations(df, similarity):
    """ Return the top 5 most similar products """
    # Get the pairwsie similarity scores of all clothes with that one (index, value)
    sim_scores = list(enumerate(similarity[0]))
    
    # Sort the clothes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 5 most similar clothes
    sim_scores = sim_scores[0:5]
    print(sim_scores)
    # Get the clothes indices
    cloth_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar products
    return df['image'].iloc[cloth_indices]

df_embeddings = get_embeddings(styles_df, MODEL)
df_embeddings.to_csv('styles_resnet50.csv')


file_number = "10034.jpg"
url=path+'/images/'+file_number
a = plt.imread(url)
plt.imshow(a)


sample_image = predict(MODEL, file_number)
sample_image.shape

df_sample_image = pd.DataFrame(sample_image)  
print(df_sample_image)

sample_similarity = cosine_similarity(df_sample_image, df_embeddings)
print(sample_similarity)

sample_similarity_norm = normalize_sim(sample_similarity)
sample_similarity_norm.shape

recommendation = get_recommendations(styles_df, sample_similarity_norm)
recommendation_list = recommendation.to_list()

#recommended images
plt.figure(figsize=(20,20))
j=0
for i in recommendation_list:
    plt.subplot(6, 10, j+1)
    cloth_img =  mpimg.imread(path + 'images/'+ i)
    plt.imshow(cloth_img)
    plt.axis("off")
    j+=1
plt.title("Recommended images",loc='left')
plt.subplots_adjust(wspace=-0.5, hspace=1)
plt.show()