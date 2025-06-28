# Automated Upper Endoscopy Auditing of Procedure Times: Temporal Multiclass Analysis
<img src="figures/Pipeline.jpg" alt="Pipeline" width="700">

Welcome to the official repository for our MICCAI 2025 paper. Here, you'll find scripts, datasets, and models essential for our research. 🚀

📊 Data
Summary: 
🔗 **Dataset:** [Figshare](https://doi.org/10.6084/m9.figshare.27308133)
🔗 **Code:** [GitHub](https://github.com/Endo2025/EndoReview.git)

This section provides an overview of the datasets used in our study 📌.
- 📼 Videoendoscopies for Organ Classification: 237 MP4 videos from 233 patients (∼1.6 million frames).
- 📊 Stomach Site Sequences: 4,729 short sequences for analysis.

📂 For more details: Check out the [data.md](data.md) file for a comprehensive guide on data organization and preprocessing steps.

## 🎯 Multi-Frame Embedding

Embedding Methods:
- 1️⃣ ViT’s Patch-Based Linear Projection (16×16×3)
- 2️⃣ ConvNeXt-Tiny Pretrained on ImageNet
- 3️⃣ ConvNeXt-Tiny Pretrained on Endoscopy

📂 Learn More: Check the [features.md](features.md) file for detailed embedding representations of each videoendoscopy and sequence feature extraction.

## 🏷️ Organ Classification

Summary of Experiments

🔍 Spatial-Based Classification
- 1️⃣ ViT’s Patch-Based Linear Projection + MLP
- 2️⃣ ConvNeXt-Tiny Pretrained on ImageNet + MLP
- 3️⃣ ConvNeXt-Tiny Pretrained on Endoscopy + MLP

    | Embedding            | Resolution | Precision | Recall | F1    | MCC   |Download                                                                     |
    |:------------------:  |:----------:|:---------:|:------:|:-----:|:-----:|:---------------------------------------------------------------------------:|
    | Linear Projection    | 1 frame    | 49.74     | 72.21  | 54.48 | 48.86 | [Download](https://drive.google.com/uc?id=1OWiScIp0P6q37BunhPj6Q3C6ZIQZxDXA)|
    | ConvNeXt (ImageNet)  | 1 frame    | 62.78     | 85.15  | 70.37 | 68.13 | [Download](https://drive.google.com/uc?id=1A0h6V5HLpqyoaMzdrLFH32ksbJr6J9VF)|
    | ConvNeXt (Endoscopy) | 1 frame    | 64.55     | 87.06  | 71.68 | 70.38 | [Download](https://drive.google.com/uc?id=1vVVVwEFlAPBLpiIjoQ5eJtY5rYg8fbH4)|

🔄 Multi-Frame-Based Classification

Summary of Experiments

⏳ Temporal-Based Classification with Attention Mechanisms

- 1️⃣ One Attention Layer initialized with Random Weights
- 2️⃣ ViT-Base initialized with Random Weights
- 3️⃣ ViT-Base initialized with ImageNet Pretraining

<!--
📂 For more details, refer to the [organclassification.md](organclassification.md) file.
-->
📂 The trained models are available. The training scripts and organ classification labels will be released upon the official publication of the paper.

- 1️⃣ One Attention Layer initialized with Random Weights

    | time   | Precision| Recall    | F1     | MCC    |Download                                                                     |
    |--------|----------|-----------|--------|--------|--------|
    | 1.0 sec| 74.57    | 85.85     | 79.02  | 76.67  | [Download](https://drive.google.com/uc?id=1ubouZ8vKvkGIpRnLPuTlla0sJR6nuqlo)|
    | 3.0 sec| 82.90    | 88.90     | 85.54  | 84.02  | [Download](https://drive.google.com/uc?id=1oFOZP8CQwqeOszbBOFLxb-Amj3ZAApkI)|
    | 5.0 sec| 83.94    | 88.39     | 85.91  | 84.29  | [Download](https://drive.google.com/uc?id=1qcwzhtbZmRzodTHNg1FAH0xQi-A1Y3g0)|
    | 9.0 sec| 85.80    | 86.17     | 85.16  | 84.26  | [Download](https://drive.google.com/uc?id=1MjIOVvNUt5sESBq9o7A-2RZ9TGmEgyrG)|
    | 13.1sec| 86.56    | 84.75     | 84.54  | 83.95  | [Download](https://drive.google.com/uc?id=1Oi8RfUBZkTSVf8y3AO_P5MJiuzMp_Y3N)|

- 2️⃣ ViT-Base initialized with Random Weights

    | time   | Precision| Recall    | F1     | MCC    |Download                                                                     |
    |--------|----------|-----------|--------|--------|--------|
    | 1.0 sec| 70.60    | 86.10     | 76.85  | 74.64  | [Download](https://drive.google.com/uc?id=1bEW2ik5KoMI1pCVGB2XJKkoQYVJMKHBC)|
    | 3.0 sec| 78.54    | 89.47     | 83.19  | 80.80  | [Download](https://drive.google.com/uc?id=1ZNdHo3Us9C_vfaIN7RYvgVzzZMb6KCBl)|
    | 5.0 sec| 80.47    | 89.86     | 84.53  | 82.56  | [Download](https://drive.google.com/uc?id=1HOJUPkxxUjn3HUT7xGkNH9NTSEQ7698d)|
    | 9.0 sec| 77.56    | 87.52     | 80.98  | 78.96  | [Download](https://drive.google.com/uc?id=11b7UYdeW3tiedns26CFF9KyZ2ckaWPIr)|
    | 13.1sec| 77.31    | **90.33** | 82.72  | 80.11  | [Download](https://drive.google.com/uc?id=1w0ee4h7Lmq8Dj3wxMfIvvydaQefT7AoC)|

- 3️⃣ ViT-Base initialized with ImageNet Pretraining

    | time   | Precision| Recall    | F1       | MCC      |Download                                                                     |
    |--------|----------|-----------|----------|----------|--------|
    | 1.0 sec| 82.24    | 88.10     | 84.96    | 83.08    | [Download](https://drive.google.com/uc?id=1xnwvKi66rQiAAyv7F3-yTWSFo46kfPr3)|
    | 3.0 sec| 89.74    | 89.14     | 89.14    | 87.85    | [Download](https://drive.google.com/uc?id=1T8aFFWa8NYvUfMQhQEsBjTdpGA_73hyv)|
    | 5.0 sec| 91.03    | 90.29     | 90.29    | 89.62    | [Download](https://drive.google.com/uc?id=1sVsYEbqIZALPG9OArPQcqer27l64P3ss)|
    | 9.0 sec| **92.03**| 90.42     | **90.42**| **89.94**| [Download](https://drive.google.com/uc?id=1mnUBkxdWrNNpR_Yg_V0eWIzN6AyPfmsV)|
    | 13.1sec| 89.87    | 88.64     | 88.64    | 88.19    | [Download](https://drive.google.com/uc?id=1mvtLAkSe8h6STXE4LSPcF47_FklFDzAw)|   

## 🏥 Stomach Sites Classification

**Summary of Experiments**
- 🔬 **Selected Embedding:** ConvNeXt-Tiny Pretrained on Endoscopy
- ⏳ **Temporal-Based Evaluation** using different time intervals:
  - 1️⃣ **ViT-Base initialized with Organ Pretraining – 3.0 sec**
  - 2️⃣ **ViT-Base initialized with Organ Pretraining – 9.0 sec**
  - 3️⃣ **ViT-Base initialized with Organ Pretraining – 13.1 sec**
<!--
📂 For a detailed breakdown, refer to the [stomachsiteclassification.md](stomachsiteclassification.md) file.
-->
📂 The trained models are available. However, the training scripts will be released after the paper is officially published.

- 1️⃣ **ViT-Base initialized with Organ Pretraining – 3.0 sec**
    | time   | Precision | Recall    | F1        | MCC       |Download                                                                     |
    |--------|-----------|-----------|-----------|-----------|--------|
    | 1.0 sec| 83.38±0.46| 82.66±0.05| 81.62±0.49| 82.45±0.40| [Download](https://drive.google.com/uc?id=19vaVy1zTSAyPPh5HGYpnuo9ZJ5VxzLAv)|
    | 2.0 sec| 85.80±0.44| 84.99±0.48| 84.39±0.47| 85.41±0.42| [Download](https://drive.google.com/uc?id=1nj6AZaYCsJBPcRxp2W6XGKXIJHd73kwa)|
    | 3.0 sec| 83.87±0.44| 83.64±0.47| 82.38±0.47| 83.22±0.42| [Download](https://drive.google.com/uc?id=1ZJrPqdOvvLtGj2_v2nU5JZFcRnfLgyKI)|
    | 5.0 sec| 86.02±0.40| 86.04±0.42| 84.96±0.41| 86.04±0.34| [Download](https://drive.google.com/uc?id=1fvfQgaH5VYq2Kf2qcHaZeYrBDanj7SEO)|
    | 6.0 sec| 86.63±0.42| 86.18±0.44| 85.47±0.43| 86.26±0.37| [Download](https://drive.google.com/uc?id=1iJpAbhmo4zcxKx3UhxrKB5LFu1zCp1LU)|
    | 7.0 sec| 87.66±0.38| 87.30±0.40| 86.45±0.39| 87.38±0.33| [Download](https://drive.google.com/uc?id=1H2qAC8JlMgN2R0zexKox80qqgjHrsKhy)|
    
- 2️⃣ **ViT-Base initialized with Organ Pretraining – 9.0 sec**
    | time   | Precision | Recall    | F1        | MCC       |Download                                                                     |
    |--------|-----------|-----------|-----------|-----------|--------|
    | 1.0 sec| 84.20±0.46| 83.43±0.47| 82.71±0.48| 83.46±0.43| [Download](https://drive.google.com/uc?id=1IG7-yhA_RZa6Btn97SqR_JqwE0IcpsO6)|
    | 2.0 sec| 85.95±0.41| 85.94±0.41| 85.02±0.42| 85.98±0.36| [Download](https://drive.google.com/uc?id=1745fOvH91A0_XC56UCXeZCkp8i2YWM5N)|
    | 3.0 sec| 85.08±0.44| 84.01±0.44| 83.02±0.46| 83.94±0.40| [Download](https://drive.google.com/uc?id=1SkiSG1TwJWFraxaikPu8z4tN8BX1sfqB)|
    | 5.0 sec| 87.48±0.39| 87.18±0.41| 86.26±0.41| 87.44±0.34| [Download](https://drive.google.com/uc?id=1h8c90VADPJz5nPWhm6L6COmr_Xavtazl)|
    | 6.0 sec| 87.03±0.34| 86.27±0.39| 85.47±0.37| 86.21±0.34| [Download](https://drive.google.com/uc?id=1k5tVvsX6pYZftWlrdnZGCda3nzQ2BC-3)|
    | 7.0 sec| 84.90±0.43| 84.91±0.44| 83.39±0.46| 84.71±0.38| [Download](https://drive.google.com/uc?id=1Eca_keTxZlu8cRNBMA7zVy1Bp8fOQMct)|

- 3️⃣ **ViT-Base initialized with Organ Pretraining – 13.1 sec**
    | time   | Precision| Recall    | F1     | MCC       |Download                                                                     |
    |--------|---------------|---------------|-----------|-----------|--------|
    | 1.0 sec| 83.21±0.47    | 81.87±0.49    | 80.97±0.48| 82.36±0.39| [Download](https://drive.google.com/uc?id=1511WD-WPRk5EeKF_LXlrE2ItExWVEkCm)|
    | 2.0 sec| 86.08±0.40    | 85.49±0.43    | 84.67±0.42| 85.84±0.35| [Download](https://drive.google.com/uc?id=1DlbFexhuM4FivrMGqT6xpI7fbCfo5rQ4)|
    | 3.0 sec| 86.14±0.37    | 85.21±0.45    | 84.56±0.42| 85.26±0.40| [Download](https://drive.google.com/uc?id=1mX6HUF0tmYLwaavH6Dg9zdLSeNTWVgBb)|
    | 5.0 sec| 85.61±0.44    | 84.64±0.47    | 83.65±0.46| 84.69±0.39| [Download](https://drive.google.com/uc?id=1mRYxtqeiVxQUYXH9KwRKwpW8sfLetMiK)|
    | 6.0 sec| 87.50±0.37    | 87.22±0.42    | 86.30±0.41| 87.12±0.35| [Download](https://drive.google.com/uc?id=1y978rZkoQPkPN-n6iDxHMYWzf3hl72bS)|
    | 7.0 sec| **88.37±0.36**| **87.82±0.37**| **87.03±0.39**| 87.79±0.29| [Download](https://drive.google.com/uc?id=1GsuP4CroKvStZGm39JbVtRrlZx33FcuV)|

## 📊 Report Quality Indicators

1️⃣ **Indicator 1: Organ-Specific Exploration Time 📖**

🩺 **Protocol Reference:**
📖 Bisschops, Raf, et al. "Performance measures for upper gastrointestinal endoscopy: a European Society of Gastrointestinal Endoscopy (ESGE) quality improvement initiative." Endoscopy 48.09 (2016): 843-864.

This metric evaluates the duration of exploration for each organ during the endoscopic procedure, ensuring adherence to standardized protocols.

| Patients | Procedure | Pharynx | Esophagus | Stomach | Duodenum |
|:---------:|:-----------:|:--------:|:----------:|:--------:|:---------:|
| 15 | 9:22±4:17 | 0:13±0:17 | 0:54±0:38 | 7:17±2:54 | 0:56±1:19 |

2) **Indicator 2:** Stomach Sites Duration (Protocol SSS: 📖). L: lesser curvature, A: anterior
wall, G: greater curvature, P: posterior wall, and SSS: systematic screening protocol
for the stomach.

📖 Yao, Kenshi. "The endoscopic diagnosis of early gastric cancer." Annals of Gastroenterology: Quarterly Publication of the Hellenic Society of Gastroenterology 26.1 (2013): 11.

| Region | Site | Time | Region | Site | Time |
|:---------:|:----:|:--------:|:---------:|:----:|:--------:|
| Antrum Antegrade | A1 | 0:21±0:10 | Lower Body Antegrade | A2 | 0:11±0:06 |
| | L1 | 0:29±0:27 | | L2 | 0:11±0:06 |
| | P1 | 0:19±0:13 | | P2 | 0:15±0:12 |
| | G1 | 0:36±0:19 | | G2 | 0:34±0:36 |
| Middle Body Antegrade | A3 | 0:08±0:06 | Fundus Cardia Reflex | A4 | 0:05±0:04 |
| | L3 | 0:07±0:06 | | L4 | 0:06±0:04 |
| | P3 | 0:11±0:08 | | P4 | 0:06±0:05 |
| | G3 | 0:24±0:17 | | G4 | 0:09±0:07 |
| Middle Body Reflex | A5 | 0:05±0:05 | Incisura Reflex | A6 | 0:11±0:09 |
| | L5 | 0:10±0:08 | | L6 | 0:11±0:11 |
| | P5 | 0:05±0:03 | | P6 | 0:10±0:09 |


<img src="figures/Quality_Indicator.jpg" alt="QI" width="700">

## 🔨 Installation
Please refer to the [libraries.md](libraries.md) file for detailed installation instructions.



## 📓 Notebooks
`predict_example.ipynb`:  Use this notebook to run sequence classification tasks for inference.

Note 🗈:  To run this code in Google Colab, click the logo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12zrP3bqw8FftpLJu1uWQ2nvvBKH6iJ_2)



