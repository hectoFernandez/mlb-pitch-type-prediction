# MLB Pitch Type Prediction using Machine Learning

## Overview
This project focuses on predicting the type of pitch thrown in Major League Baseball (MLB) using supervised machine learning models. The dataset contains pitch-by-pitch Statcast data from MLB postseason games, including physical characteristics of the pitch, contextual game information and player data. The objective is to train and compare several machine learning models to determine which model performs best at classifying pitch types.

## Dataset
The dataset used in this project is the MLB Statcast Postseason Pitch-by-Pitch Data. Each observation represents a single pitch and includes information about pitch physics (velocity, spin, trajectory), game context (balls, strikes, outs, runners on base) and player characteristics (pitcher handedness, batter stance). The original dataset contains 14,096 pitches and 95 variables. To improve computational efficiency and focus on relevant features, a subset of 5,000 observations and 29 relevant variables was used.

## Target Variable
The model predicts the variable `pitch_type`. Possible pitch types include FF (Four-Seam Fastball), SL (Slider), CH (Changeup), CU (Curveball), SI (Sinker), FC (Cutter), KC (Knuckle Curve), FS (Splitter), ST (Sweeper) and SV (Slurve).

## Machine Learning Models
Several supervised learning models were implemented and compared.

K-Nearest Neighbors (KNN): Hyperparameters tested included the number of neighbors (from 1 to 20) and the distance metric (Euclidean and Manhattan). The best configuration was k = 1 with the Manhattan distance metric, achieving an F1-macro score of 0.794 in cross-validation.

Support Vector Machine (SVM): Hyperparameters tested included C values of 0.1, 1, 10 and 100, kernels (linear and RBF), gamma values (scale and auto) and class weights. The best configuration was C = 10, kernel = RBF and gamma = scale, achieving an F1-macro score of 0.814 in cross-validation.

Logistic Regression: Hyperparameters tested included C values between 0.001 and 1000, solvers (lbfgs and newton-cg) and class weights. The best configuration was C = 300 with the lbfgs solver and no class weighting, achieving an F1-macro score of 0.674.

Decision Tree: Hyperparameters tested included max_depth, min_samples_leaf, min_samples_split, max_features, criterion (gini or entropy) and the pruning parameter ccp_alpha. The best model achieved an F1-macro score of 0.617.

## Best Model
After comparing all models, the best performing model was the Support Vector Machine (SVM). The performance on the test dataset was: Accuracy = 0.879, Precision = 0.849, Recall = 0.748 and F1-score = 0.767.

## Machine Learning Pipeline
The project follows a typical machine learning workflow that includes data preprocessing, feature selection, train/test split (80/20), feature scaling using StandardScaler, model training, hyperparameter tuning using GridSearchCV, cross-validation with 10 folds and final evaluation on the test dataset.

## Technologies Used
The project was implemented using Python with the following libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn and Jupyter Notebook.

## Repository Structure
data/  
Data_MLB_2025_StatcastPostseason_PitchByPitch.csv  

notebooks/  
mlb.ipynb  

reports/  
Memoria_AA.pdf  
Presentacion_AA.pdf  

README.md

## Authors
Héctor Fernández Cano  
Diego José García Callejas  
Data Science and Artificial Intelligence  
Universidad Politécnica de Madrid
