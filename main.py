import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt
from skimage.io import imshow
from sklearn.model_selection import train_test_split
from satellite_cls_model import sat_class_model
import pickle
from predict_class import predict_class
from compute_metric_score import compute_metrics

# from google.colab import drive
# drive.mount('/content/drive')

X = pd.read_csv('/content/drive/MyDrive/NNDL_Project/X_test_sat4.csv') 
Y = pd.read_csv('/content/drive/MyDrive/NNDL_Project/y_test_sat4.csv')

X = np.array(X) 
Y = np.array(Y)

X = X.reshape([99999, 28, 28, 4]).astype(float) 

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0) 

x_train = x_train / 255
x_test = x_test / 255

img_no = random.randint(0, 79999) 
imshow(np.squeeze(x_train[img_no, :, :, 0:3]).astype(float)) 
plt.show()

model = sat_class_model(num_classes = 4)
model_pred = model.fit(x_train, y_train, batch_size = 64, epochs = 20, verbose = 1, validation_split = 0.20)

file_name = 'sat_cls_model.sav'
pickle.dump(model, open(file_name, 'wb'))

img_no = random.randint(0, 19999)
imshow(np.squeeze(x_test[img_no, :, :, 0:3]).astype(float)) 
plt.show()

preds = predict_class(file_name, x_test, img_no, y_test)

compute_metrics(y_test, preds)
