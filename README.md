# 汽車種類辨識 (Vehicle Type Recognition)

比較多種傳統特徵（HOG、彩色直方圖）搭配多種分類器（SVM、KNN、決策樹、隨機森林、Gradient Boosting Tree），以及 CNN（VGG16）特徵，對 Car / Bus / Motorcycle / Truck 四類車輛影像進行分類，並比較各方法準確率與混淆矩陣。

## 技術棧
- Python, OpenCV, scikit-image（HOG 特徵）
- scikit-learn（SVM、KNN、Decision Tree、Random Forest、GBT）
- TensorFlow / Keras（VGG16 特徵提取）

## 主要檔案
- `main.py`：HOG + CNN(VGG16) 特徵分別搭配 SVM / 隨機森林分類
- `mainx.py`：彩色直方圖特徵分類比較
- `HOG_multi.py`：HOG 特徵搭配多種分類器比較
- `HOG+histogram/`、`HOG_multi/`、`color_histogram_multi(XX)/`：各方法的準確率與混淆矩陣結果圖

## 資料集
- `Vehicle_Type_Recognition/`：Car / Bus / Motorcycle / Truck 四類車輛影像資料集（常見於 Kaggle 公開資料集），因檔案數量龐大（約 170MB），未納入本 repo，執行前請自行下載並放置於同名資料夾，分類子資料夾需對應程式中的 `categories = ['car', 'bus', 'motorcycle', 'truck']`。

## 執行方式
```bash
pip install opencv-python scikit-image scikit-learn tensorflow numpy seaborn matplotlib
python main.py
```
