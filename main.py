import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt
from skimage.io import imshow
import pickle

from satellite_cls_model import sat_class_model
from predict_class import predict_class
from compute_metric_score import compute_metrics

x_train = pd.read_csv('/kaggle/input/X_train_sat4.csv')
y_train = pd.read_csv('/kaggle/input/y_train_sat4.csv')

x_test = pd.read_csv('/kaggle/input/X_test_sat4.csv')
y_test = pd.read_csv('/kaggle/input/y_test_sat4.csv')

x_train = np.array(x_train) 
y_train = np.array(y_train)

x_test = np.array(x_test) 
y_test = np.array(y_test)

x_train = x_train.reshape([399999, 28, 28, 4]).astype(float) 
x_test = x_test.reshape([99999, 28, 28, 4]).astype(float) 

x_train = x_train / 255
x_test = x_test / 255

img_no = random.randint(0, 399999) 
imshow(np.squeeze(x_train[img_no, :, :, 0:3]).astype(float)) 
plt.show()

model = sat_class_model(num_classes = 4)
model_pred = model.fit(x_train, y_train, batch_size = 64, epochs = 30, verbose = 1, validation_split = 0.20)

file_name = '/kaggle/working/sat_cls_model.sav'
pickle.dump(model, open(file_name, 'wb'))

img_no = random.randint(0, 99999)
imshow(np.squeeze(x_test[img_no, :, :, 0:3]).astype(float)) 
plt.show()

preds = predict_class(file_name, x_test, img_no, y_test)

compute_metrics(y_test, preds)
