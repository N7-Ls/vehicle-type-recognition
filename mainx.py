# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:09:17 2024

@author: 言
"""
import os
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# #%% SIFT特徵提取
# def extract_sift_features(image):
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     if descriptors is None:
#         return np.zeros(128)
#     return descriptors.flatten()

#%% 彩色直方圖特徵提取
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

#%% 載入資料集並進行特徵提取
def load_data(dataset_path):
    X = []
    y = []
    classes = os.listdir(dataset_path)
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hog_features = hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')
            # lbp_features = local_binary_pattern(gray_image, P=8, R=1).flatten() X
            # sift_features = extract_sift_features(gray_image) X
            color_histogram_features = extract_color_histogram(image)
            combined_features = np.hstack([hog_features,color_histogram_features])#[hog_features, lbp_features, sift_features, color_histogram_features]
            X.append(combined_features)
            y.append(class_name)
            print(f"Processed {image_path}, Combined features shape: {combined_features.shape}")
    return X, y

#%% 指定資料集路徑
dataset_path = "Vehicle_Type_Recognition/Dataset"

# 載入資料
X, y = load_data(dataset_path)

#%% 找出最大的特徵向量長度
max_length = max(len(f) for f in X)
print(f"Max feature length: {max_length}")

# 確保特徵向量長度一致
X_padded = np.array([np.pad(f, (0, max_length - len(f)), mode='constant') for f in X])

#%% 將標籤轉換為NumPy數組
y = np.array(y)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=7)

#%% 特徵縮放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% 定義模型
models = {
    "Linear SVM": SVC(kernel='linear', random_state=7),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=6, p=2, weights='uniform'),
    "Decision Tree": DecisionTreeClassifier(random_state=7, min_samples_split=15),
    "Random Forest": RandomForestClassifier(n_estimators=180, random_state=7, max_depth=20, min_samples_split=15),
    # "Gradient Boosting Tree": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=7)
}

#%% 訓練和評估模型
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy}")
    print(classification_report(y_test, predictions))
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_test, predictions)
    print(f"{model_name} Confusion Matrix:\n{cm}")
    
    # 可視化混淆矩陣
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
