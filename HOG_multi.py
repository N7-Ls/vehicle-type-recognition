# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:05:41 2024

@author: 言
"""
import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 載入資料集並進行特徵提取
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
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            hog_features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')
            X.append(hog_features)
            y.append(class_name)
            print(f"Processed {image_path}, HOG features shape: {hog_features.shape}")
    return X, y

# 指定資料集路徑
dataset_path = "Vehicle_Type_Recognition/Dataset"

# 載入資料
X, y = load_data(dataset_path)

#%% 找出最大的特徵向量長度
max_length = max(len(f) for f in X)
print(f"Max feature length: {max_length}")

# 確保特徵向量長度一致
X_padded = np.array([np.pad(f, (0, max_length - len(f)), mode='constant') for f in X])

# 將標籤轉換為NumPy數組
y = np.array(y)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# 特徵縮放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% 定義模型
models = {
    # "Linear SVM": SVC(kernel='linear', random_state=42),
    # "Polynomial SVM": SVC(kernel='poly', degree=3, random_state=42),
    # "RBF SVM": SVC(kernel='rbf', random_state=42),
    # "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    # "Decision Tree": DecisionTreeClassifier(random_state=42),
    # "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting Tree": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
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
