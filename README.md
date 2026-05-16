# MIA Final Project: BraTS2020 腦瘤分析系統

這是一個基於深度學習的醫學影像分析 (MIA, Medical Image Analysis) 專案，主要針對 BraTS2020 腦瘤 MRI 影像數據集進行多任務學習與可解釋性人工智慧 (XAI) 的分析。

## 資料來源 (Data Source)

本專案使用 Kaggle 上的 BraTS2020 訓練資料集。若要在本地端執行，您可以使用 `kagglehub` 自動下載，或是手動從以下連結下載並放置於專案目錄中：

=> **[Kaggle: BraTS2020 Training Data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data?resource=download)**

資料集包含四種 MRI 模態影像：T1、T1Gd、T2、FLAIR，以及相關的臨床存活天數與病患資訊。

## 專案架構與核心功能

專案分為兩個主要的執行腳本，分別對應不同的訓練策略與模型架構：

### 1. `main.py` - 多任務 Cross-Modal 模型與 Baseline 比較
這是專案的主要流程，實作了一個強大的**多任務 3D 網路**，並提供 Baseline 模型的比較功能。
- **主要模型 (Proposed):** Cross-Modal Swin-Transformer 3D
  - 透過 Cross-Modal Attention 融合 T1/T1Gd 與 T2/FLAIR 模態的特徵。
  - **多任務學習 (Multi-task Learning):** 
    1. 腫瘤子區域分割 (Segmentation) - 註：預設使用影像強度規則產生的 Pseudo Label。
    2. 存活天數回歸預測 (Survival Days Regression) - 結合臨床年齡資料。
    3. 優先級分類 (Priority Classification)。
- **Baseline 模型:** 提供 3D-UNet 與 ResNet3D 作為效能比較的基準。
- **輸出:** 訓練曲線 (Loss, Dice, MAE, Accuracy)、各類別 Dice 分數圖表以及模態預覽圖。

### 2. `xai.py` - 官方標註分割與可解釋性 AI (XAI)
此腳本專注於使用**官方提供的真實腫瘤標註 (Official Mask)** 進行訓練，並強調模型決策的可解釋性。
- **模型架構:**
  - Cross-Modal Swin Segmentation (純分割版本)
  - Attention 3D U-Net (具備 Attention Gate，適合 XAI 分析)
- **可解釋性 AI 功能:**
  - 產生 Grad-CAM++ 熱圖 (Heatmaps)。
  - 影像重疊對比：黃色輪廓代表官方真實標註，青色輪廓代表模型預測，幫助直觀檢視模型預測結果的準確度。
- **評估指標:** 主要評估 Tumor Region 的 IoU 與 Dice 分數。

## 安裝與環境需求

建議使用 Python 3.8+ 與 PyTorch。主要的依賴套件包含：
```bash
pip install torch torchvision torchaudio
pip install h5py numpy pandas matplotlib scikit-learn tqdm opencv-python
pip install kagglehub  # 用於自動下載資料集
```

## 使用方法

### 執行多任務模型 (main.py)
```bash
python main.py
```
> 註：3D 影像處理需要較大的 GPU 記憶體。若遇到 Out of Memory (OOM) 錯誤，請在腳本或參數中調小 `crop_d`、`crop_hw` 或減少 `batch_size`。

### 執行官方標註分割與 XAI 分析 (xai.py)
```bash
python xai.py --download-kaggle --epochs 10
```

## 注意事項
- `main.py` 預設的分割訓練是基於強度規則生成的 Pseudo Label，若需正式論文實驗的 Segmentation，建議參考 `xai.py` 讀取官方真實的 Mask 進行訓練。
- 專案中的 `.h5` 影像預期已經過預處理 (例如格式轉換與切片)。
