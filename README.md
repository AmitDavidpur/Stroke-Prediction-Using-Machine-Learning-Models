# Stroke Prediction Using Machine Learning Models
Data science and advanced Python concepts workshop for Neuroscience 

# Project Description
Stroke has a serious impact on individuals and healthcare systems, making early prediction crucial. With the growing use of technology in medicine, electronic health records (EHR) provide valuable data for improving diagnosis and patient management.

In this project, we replicate a research study that applied machine learning models to predict strokes based on EHR data. Instead of identifying key predictive factors, our goal is to implement and test the same models used in the original study. By doing so, we aim to verify the reported results and evaluate the effectiveness of these models.

This study is important because it demonstrates how machine learning can enhance stroke prediction. If the models perform well, they could contribute to better decision-making in healthcare.

# Primary Article
https://www.sciencedirect.com/science/article/pii/S2772442522000090

# Project Data
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

# Folder & Module Structure
Final_workshop_New/  
│── src/
│   ├── Preprocessing/  # Data exploration, missing value handling, encoding 
│   │   ├── preprocessing.py
│   │   ├── visualization.py  
│   ├── FeatureAnalysis/  # PCA, data balancing, feature contributions
│   │   ├── FeatureAnalysis.py  
│   │   ├── balancing.py 
│   │   ├── visualization.py   
│   ├── Models.py
│   ├── pipeline.py # Execute the workflow
│   ├── config.py
│── tests/  
│   ├── test_preprocessing.py  # Unit tests for preprocessing 
│── data/  # Original dataset  
│── plots/  # visualizations
│── ModelsResultsFiles/  # Saved model results and evaluation metrics
│── main.py   
│── pyproject.toml  # Project dependencies and configurations  
│── README.md  # Project overview and documentation
![image](https://github.com/user-attachments/assets/236d1dff-e517-4385-96f0-d0fc1377c000)


# Key stages
Data Import: Load the dataset -> 

Preprocessing: Handle missing values, encode categorical features, and perform initial data exploration. -> 

Visualizations:  Plot distributions for categorical and numerical columns to visualize data characteristics. -> 

Feature Analysis and Balancing: Perform PCA to reduce dimensionality, analyze feature contributions, and balance the dataset. -> 

Models:Train machine learning models using both full features and PCA-reduced features, followed by hyperparameter tuning through grid search. -> 

Results: Save  the model performance results, comparing them across different feature sets, and store them in CSV files.

# Explanations of Key Parameters
Categorical Columns: Columns containing categorical data

Numerical Columns: Columns containing numerical data

Encoding Dictionary for Categorical Columns: The categorical columns are encoded to numerical values for model compatibility.

Features List: The list of features used as input for the models.

Models: The models defined and used for training the dataset.

Grid Search Parameters: Parameters for each model used to tune and optimize the models during training.


# To run the project follow this commands:
#install Virtualenv 

pip install virtualenv

#create virtual environment:

python -m venv venv

#activate virtual environment

.\venv\Scripts\activate

#update venv's python package-installer (pip) to its latest version

python.exe -m pip install --upgrade pip

#install projects packages

pip install -e .

#install dev packages 

pip install -e .[dev]

