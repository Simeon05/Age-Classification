# Age-Classification


📚 Project Overview
This project focuses on age estimation from facial images using Deep Learning models.
It involves:

Classification (predicting age categories: Child, Teenager, Adult, Senior)

Regression (predicting actual age)

Transfer Learning (with a custom autoencoder and MobileNet)

Bias Analysis (evaluating gender bias impact)

📂 Repository Structure
Notebooks/ → Code notebooks for data preparation, modeling, and evaluation.

Models/ → Best trained models saved for reuse.

Dataset/ → Processed and cleaned dataset used for training and evaluation.

🚀 Project Highlights
Part 1: Simple Age Estimator
Data Loading: Images loaded using keras.utils.image_dataset_from_directory.

Preprocessing:

Valid image formats ensured (JPG, JPEG, BMP, PNG)

Normalization (pixel scaling [0,1])

Age labels mapped into 4 categories.

Tasks:

Classification: Predicts age category.

Regression: Predicts exact age.

Model Architectures:

5 Classification CNNs (Model_1 best with 81% accuracy).

6 Regression CNNs (Model_reg_skip best using skip connections).

Technical Challenges Solved:

Label mapping

Loss function mismatch

Overfitting and underfitting

Part 2: Transfer Learning with Pretrained Encoder
Encoder Pretraining: Custom autoencoder trained on Block 1.

Classification Models:

Base Model

Tuned Models 1, 2

Tuned Model 3: Best performance after unfreezing the encoder.

Benchmark Model: Fine-tuned MobileNet (best overall accuracy).

Highlights:

Fine-tuning improved feature learning.

Skip connections and deeper heads enhanced performance.

Part 3: Bias Analysis - Gender Segregation
Gender-based Datasets:

Male-only dataset

Mixed-gender dataset

Models: Identical CNNs trained separately on each subset.

Findings:

Male-only model achieved higher accuracy (85.7%) but showed signs of overfitting.

Mixed-gender model had better generalization (higher F1-score: 0.9720).

📈 Key Results

Model	Accuracy	Precision	Recall	F1-Score
Model_1 (Simple CNN)	81.08%	97.63%	-	-
Tuned Model 3 (Autoencoder)	Best performance among homegrown models			
MobileNet (Pretrained)	Best overall accuracy			
Male-Only Model	85.71%	95.39%	95.39%	95.39%
Mixed-Gender Model	78.57%	96.89%	97.50%	97.20%
⚙️ Technologies Used
TensorFlow

Keras

Python 3.9+

Google Colab

📜 Acknowledgments
Pre-trained gender classification model: [Caffe Model from GitHub]

Pre-trained MobileNet from TensorFlow Hub

Face-Age Dataset

📝 Declaration
I confirm that the document and related code here is all my own work and that I did not engage in unfair practice in any way.

