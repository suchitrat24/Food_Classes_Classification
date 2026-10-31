# Food Classes Classification Using CNN

## Project Overview

This project builds a deep learning model that looks at a food photo and predicts which of **101 types of food** it shows, using Convolutional Neural Networks (CNNs) and transfer learning.

The project was completed as part of the **DataMites AI Engineer Program (Internship Project)**.

The primary objective was to analyze the food images (size, quality, colour), understand what makes the classes easy or hard to separate, build and compare several CNN models on **100,990 real-life food photos**, and give practical suggestions on how to use the model in a food app.

## Project Objectives

- Perform image quality checks and exploratory data analysis (EDA).
- Find and remove duplicate images, including the same photo saved under two different classes.
- Prepare all images to a fixed size for fast CNN training.
- Show why simple colour and size features are not enough to recognise food.
- Build and compare a CNN trained from scratch with pretrained CNNs (transfer learning).
- Evaluate ensembles and test-time augmentation.
- Provide data-driven suggestions for using the model in a food app.

## Technologies & Libraries

- Python
- Jupyter Notebook
- PyTorch
- Torchvision
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

### 1. Data Checks and Cleaning
- Loaded **101,000 JPG images in 101 food classes**, exactly 1,000 images per class (perfectly balanced).
- Checked every file: no missing values and no broken images. 3 grayscale images were converted to RGB.
- Every image has 512 pixels on its longest side. 62,206 images (61.6%) are square 512 x 512.
- Found 8 extra copies using MD5 fingerprints. 2 groups were the same photo in two different classes (a wrong label), so all their copies were dropped.
- **10 images removed** in total, leaving **100,990 images** (997 to 1,000 per class).
- Resized and centre-cropped all images to 240 x 240 once and stored them in a single array for fast training.

### 2. Exploratory Data Analysis (EDA)
- Viewed one sample photo of every class and several photos within the same class.
- Added image features: aspect ratio, mean red / green / blue, brightness, and contrast.
- Analyzed pixel colour distributions: food photos are **warm** (average red 140.5, green 113.5, blue 86.8).
- Compared classes by brightness, greenness, and file size (texture).
- Used histograms, relational plots, pair plots, line plots, bar charts, box plots, and correlation heatmaps.

### 3. Outlier Analysis
Used the Interquartile Range (IQR) method and boxplots to identify extreme values.

Key findings:
- Width, height and aspect ratio flagged 12% to 36% of images only because most images are exactly 512 px. These are normal portrait or landscape photos.
- 1,585 photos (1.6%) had brightness outside 46 to 182: dark restaurant photos and over-exposed white plates.
- Some noise was found: a completely black `breakfast_burrito` image, a menu photo in `foie_gras`, and a collage of three photos.
- All outliers were kept, because users of a real food app will also take dark or unusual photos.

### 4. Feature Analysis and Label Encoding
- Encoded the 101 food names as numbers 0 to 100 using `LabelEncoder`.
- The food class explains only **6 to 8%** of the variation in each colour, contrast or file-size feature, and less than 1% of image size.
- Because of this, the CNNs use the raw pixels (224 x 224 x 3) instead of hand-made features, with colour kept and each channel normalised using the ImageNet mean and standard deviation.

### 5. Machine Learning Model Development

This project uses supervised multi-class image classification to predict the food type in a photo.

Data was split per class (stratified) into **70% training (70,690), 5% validation (5,052) and 25% test (25,248)**.

The following models were implemented and evaluated:

1. Random Forest on 9 simple image features (baseline)
2. Custom CNN built from scratch (5 convolution blocks, 128 x 128 input)
3. MobileNetV3-Large as a frozen feature extractor, with a grid search on the last layer
4. MobileNetV3-Large fine-tuned
5. EfficientNet-B0 fine-tuned

Training used data augmentation (random crops, flips, brightness changes), cross-entropy loss with label smoothing 0.1, the AdamW optimizer, a one-cycle learning rate schedule, and mixed precision on an Intel Arc GPU.

### 6. Model Evaluation and Comparison

Models were evaluated on the held-out test set using:
- Accuracy (top-1)
- Top-5 Accuracy
- Macro F1 Score
- Training time and prediction speed

| Model | Test Accuracy | Top-5 Accuracy | Macro F1 | Train time | Speed (images/s) |
|---|---:|---:|---:|---:|---:|
| Random Forest (simple features) | 4.1% | 13.5% | 0.036 | < 1 min | - |
| Custom CNN (from scratch) | 61.3% | 84.5% | 0.610 | 20.5 min | 1,356 |
| MobileNetV3 (feature extractor) | 67.9% | 88.8% | 0.678 | 2 min | - |
| MobileNetV3 (fine-tuned) | 80.8% | 94.6% | 0.807 | 17.0 min | 909 |
| EfficientNet-B0 (fine-tuned) | 82.1% | 95.1% | 0.821 | 34.9 min | 507 |

Random guessing would give about 1.0% accuracy. Learning curves showed no overfitting, and all curves were still rising at the last epoch.

### 7. Ensembles and Test-Time Augmentation
- **Test-time augmentation (TTA):** each photo is predicted as is and flipped left-right, and the two answers are averaged.
- **Soft Voting:** average of the probabilities from the two fine-tuned CNNs (with TTA).
- **Stacking:** a Logistic Regression trained on the validation set predictions of the two CNNs.

| Model | Test Accuracy | Top-5 Accuracy | Macro F1 | Speed (images/s) |
|---|---:|---:|---:|---:|
| MobileNetV3 + TTA | 81.0% | 94.8% | 0.809 | 463 |
| **EfficientNet-B0 + TTA** | **82.5%** | **95.2%** | **0.825** | 271 |
| Soft Voting (2 CNNs + TTA) | **83.5%** | **95.6%** | 0.834 | - |
| Stacking (2 CNNs + TTA) | 80.7% | 94.4% | 0.807 | - |

Stacking did worse than both of its CNNs, because 5,052 validation photos (about 50 per class) were too few to train it without overfitting.

**Best Performing Model: EfficientNet-B0 (fine-tuned) + TTA**

EfficientNet-B0 with flip TTA is the best single model, with **82.5% accuracy, 95.2% top-5 accuracy and macro F1 0.825** on 25,248 test photos. Soft Voting is 0.9 points more accurate (83.5%), but it needs two networks for every photo. A single small model (4.1M parameters) is the better trade-off for a mobile food app, while the ensemble is a good upgrade for a server.

### 8. Using the Model in a Food App

The model's confidence is reliable, so it can decide how the app should answer:

| Confidence at least | Photos answered directly | Accuracy on those photos |
|---|---:|---:|
| 0.0 | 100.0% | 82.5% |
| 0.5 | 79.9% | 92.7% |
| 0.7 | 65.9% | 96.4% |
| 0.9 | 35.0% | 98.9% |

For photos below 0.7 confidence, the true food is in the top 3 guesses 81.1% of the time.

## Key Findings

- Simple colour and size features are not enough: a Random Forest on them reached only **4.1%** accuracy, while a CNN learning from pixels reached **61.3%**.
- Transfer learning was the key technique: fine-tuned pretrained CNNs reached **80 to 83%** in only 3 to 4 epochs.
- Fine-tuning beat a frozen feature extractor by **12.8 points** (80.8% vs 67.9%).
- The easiest classes had a unique colour or shape, such as edamame (F1 0.976) and seaweed salad (0.944). The hardest was steak (F1 0.549, test accuracy 52.0%).
- Most mistakes were between dishes that look alike: steak was predicted as filet mignon for **15.2%** of steak photos, and chocolate mousse as chocolate cake for **14.0%**.
- Compared within the same food, dark photos were **5.0 points** less accurate and worse in **74 of the 101 classes**. Bright photos were not a problem (+1.7).

## Food App Recommendations

- Show the answer directly when confidence is at least 0.7, and show the top-3 foods for the user to choose from otherwise.
- Treat look-alike dishes as groups (for example "beef steak" or "chocolate dessert"). The 12 most confused pairs cause 433 of the 4,409 test errors (9.8%).
- Warn the user when a photo is very dark and suggest using the flash or framing the plate in a square.
- Collect more and clearer photos for the weakest classes, such as steak, chocolate mousse, bread pudding, foie gras, and pork chop.
- Clean the labels by reviewing training photos where the model strongly disagrees with the label.
- Train longer, use a larger input size or bigger network, or use the Soft Voting ensemble on a server to gain more accuracy.

## Challenges and Solutions

Some of the main challenges encountered included:

- **Duplicate and wrongly labelled images:** Used MD5 fingerprints to remove exact copies and dropped photos found in two different classes.
- **Large image data:** Resized all images once into a single array so training did not have to decode 5 GB of JPG files every epoch.
- **Weak hand-made features:** Used CNNs that learn shapes and textures directly from the pixels.
- **Overfitting and noisy labels:** Used data augmentation and label smoothing.
- **Look-alike food classes:** Used top-5 accuracy, confidence thresholds, and food groups to make the model useful even when the top guess is wrong.

## Disclaimer

This project was completed for educational and portfolio purposes as part of the DataMites AI Engineer Program. The model results are based on a fixed set of food photos and should not be interpreted as a validated production food recognition system.

## Author

**Suchitra Thanumalayan**

Data Science | Machine Learning | Artificial Intelligence
