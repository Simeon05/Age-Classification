# Age-Classification

##  Project Overview

This project focuses on age prediction using deep learning models. It includes:
- **Classification**: Predicting age categories (Child, Teenager, Adult, Senior)
- **Regression**: Predicting actual numerical ages
- **Transfer Learning**: Using a pre-trained encoder and MobileNet
- **Bias Analysis**: Studying gender bias in age classification

---

## 📂 Repository Structure

- `Notebooks/` – Code notebooks for data preparation, modeling, and evaluation
- `Models/` – Saved best models
- `Dataset/` – Processed image dataset

---

## Project Details

### Part 1: Simple Age Estimator

- **Data Loading**: Used `keras.utils.image_dataset_from_directory` to load facial images.
- **Preprocessing**:
  - Image validation: Only JPG, JPEG, BMP, PNG formats retained
  - Data normalization: Pixel values scaled to [0,1]
  - Label mapping: Age mapped into 4 categories:
    - 0: Child (Age < 13)
    - 1: Teenager (13–19)
    - 2: Adult (20–59)
    - 3: Senior (60+)
- **Dataset Split**:
  - 70% Training
  - 20% Validation
  - 10% Testing

- **Models Developed**:
  - 5 CNN models for classification
    - `Model_1` achieved highest precision (0.9763) and accuracy (81.08%)
  - 6 CNN models for regression
    - `Model_reg_skip` with skip connections achieved best loss

- **Technical Challenges**:
  - Correct label mapping inside `tf.data` pipeline
  - Solving mismatch errors in loss functions
  - Dealing with overfitting and underfitting

---

### Part 2: Transfer Learning with Pretrained Encoder

- **Encoder Pretraining**:
  - Built a custom autoencoder and pretrained it on a portion of data (Block 1)
- **Transfer Learning Models**:
  - **Base Model** (simple head)
  - **Tuned Model 1** (larger dense layer with tanh activation)
  - **Tuned Model 2** (flattened encoder output + ReLU)
  - **Tuned Model 3** (unfrozen encoder + deep dense layers)

- **Best Model**:
  - **Tuned Model 3**: Unfreezing encoder improved feature adaptation and achieved highest performance.

- **Benchmark Model**:
  - **MobileNet V2** (fine-tuned on dataset)
  - Achieved slightly better overall accuracy compared to homegrown models.

---

### Part 3: Bias Analysis - Gender Segregation

- **Gender Separation**:
  - Used a pretrained Caffe model to detect and separate male-only images.
- **Training**:
  - Trained identical CNN architectures on:
    - Male-only dataset
    - Mixed-gender dataset

- **Results**:
  | Model | Accuracy | Precision | Recall | F1-Score |
  |:------|:---------|:----------|:-------|:---------|
  | Male-Only | 85.71% | 95.39% | 95.39% | 95.39% |
  | Mixed-Gender | 78.57% | 96.89% | 97.50% | 97.20% |

- **Observations**:
  - Male-only model had higher accuracy but signs of overfitting.
  - Mixed-gender model had better F1-score, suggesting better generalization.

---

## Key Takeaways

- Using **tanh** activation helped stabilize classification training.
- **Skip connections** significantly improved regression performance.
- **Transfer learning** (fine-tuning the encoder) boosted classification accuracy.
- **Pretrained MobileNet** outperformed custom models slightly.
- **Balanced datasets** are critical to mitigate bias.

---

## Technologies Used

- TensorFlow 2.x
- Keras
- Python 3.9+
- Google Colab
- Caffe (for gender model)
- Matplotlib, NumPy
