# Hotel Booking Cancellation Prediction Project

## Overview
This project aims to predict hotel booking cancellations using various machine-learning algorithms. The dataset used for this project is from Kaggle and can be found [here](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). The dataset contains information about hotel bookings, including features like customer demographics, booking details, and special requests.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)


## Dataset
The dataset used in this project is publicly available on Kaggle. It includes two hotel datasets, one for city hotels and the other for resort hotels. The dataset contains information about booking details, customer demographics, and other relevant features that can be used for predicting hotel booking cancellations.
Dataset Link :- https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

## Libraries Used
The following Python libraries were utilized in this project:
- pandas: Data manipulation and analysis
- numpy: Numerical operations and array manipulation
- seaborn: Data visualization
- folium: Interactive maps (if used)
- plotly: Interactive and dynamic visualizations
- scikit-learn: Machine learning algorithms implementation
- catboost: Gradient boosting library for categorical data (if used)

## Machine Learning Algorithms
Multiple machine learning algorithms were employed for this prediction task:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest Classifier
- Gradient Boosting
- CatBoost (if used)

## Usage
To run this project, make sure you have the required Python libraries installed. You can install them using `pip install -r requirements.txt`. Then, execute the Jupyter notebooks in the `notebooks/` directory sequentially for data exploration, preprocessing, and model training.

## Results
The performance of each machine learning algorithm is evaluated using appropriate metrics like accuracy, precision, recall, F1 score, etc. The best-performing model was random forest classifier with 86% accuracy.

## Contributing
If you want to contribute to this project, feel free to create a pull request. Any improvements, suggestions, or bug fixes are highly appreciated.



