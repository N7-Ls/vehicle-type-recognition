# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:23:17 2024

@author: 言
"""
#%%
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# 定義路徑
data_dir = 'Vehicle_Type_Recognition/Dataset'
categories = ['car', 'bus', 'motorcycle', 'truck']

# 圖片大小
IMG_SIZE = 128

# 加載數據
def load_data(data_dir, categories):
    data = []
    labels = []
    
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append(img_array)
                labels.append(class_num)
            except Exception as e:
                pass
    
    return np.array(data), np.array(labels)

data, labels = load_data(data_dir, categories)

#%% 標籤編碼
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# 切分數據集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#%% 提取HOG特徵
def extract_hog_features(images):
    hog_features = []
    for image in images:
        feature = hog(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 
                      orientations=9, pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(feature)
    return np.array(hog_features)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

#%% 提取CNN特徵
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block4_pool').output)

def extract_cnn_features(images):
    images = preprocess_input(images)
    features = model.predict(images)
    return features.reshape(len(images), -1)

X_train_cnn = extract_cnn_features(X_train)
X_test_cnn = extract_cnn_features(X_test)

#%% SVM分類
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_hog, np.argmax(y_train, axis=1))

y_pred_svm = svm_model.predict(X_test_hog)
print("SVM Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_pred_svm))

#%% 隨機森林分類
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_cnn, np.argmax(y_train, axis=1))

y_pred_rf = rf_model.predict(X_test_cnn)
print("Random Forest Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_pred_rf))
