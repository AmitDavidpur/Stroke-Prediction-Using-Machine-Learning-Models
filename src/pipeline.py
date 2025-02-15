import pandas as pd
from logger import logger
from Preprocessing.visualization import plot_categorical_data, plot_numerical_data
from Preprocessing.preprocessing import explore_data, fill_missing_bmi, encode_columns
from config import Categorical_Columns, Numeric_Columns, Encoding_Dictionary, Features, Models, ParametersForGridSearch
from FeatureAnalysis.visualization import plot_correlation_heatmap, plot_scree_plot
from FeatureAnalysis.feature_analysis import pca_analysis, split_data, pca_contribution
from FeatureAnalysis.balancing import balance_data
from models import grid_search_func, select_pca_components

def preprocess_data(filepath):
    """
    Preprocess the dataset: explore, visualize, handle missing values, and encode catecorical columns
    
    """
    
    logger.info("Starting preprocessing...")

    # Load the healthcare dataset into a pandas DataFrame 
    healthCareDataFrame = pd.read_csv(filepath, index_col='id')

    # Perform an initial exploration of the dataset
    explore_data(healthCareDataFrame, Categorical_Columns)

    # Visualize categorical column distributions
    plot_categorical_data(healthCareDataFrame, Categorical_Columns)

    # Visualize numerical column distributions 
    plot_numerical_data(healthCareDataFrame, Numeric_Columns)

    # Fill missing BMI values
    healthCareDataFrame = fill_missing_bmi(healthCareDataFrame)

    # Encode categorical columns using predefined encoding dictionary
    healthCareDataFrame = encode_columns(healthCareDataFrame, Encoding_Dictionary)

    logger.info("Preprocessing completed.")
    return healthCareDataFrame

def feature_analysis(healthCareDataFrame):
    """
    Perform PCA analysis before and after balancing.
    
    """
    logger.info("Starting feature analysis...")

    # Plot correlation heatmap to identify relationships between features  
    plot_correlation_heatmap(healthCareDataFrame)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(healthCareDataFrame, 'stroke')

    # Perform PCA analysis on the training data to reduce dimensionality 
    X_pca, pca = pca_analysis(X_train)

    # Plot the scree plot to visualize explained variance before balancing 
    plot_scree_plot(pca, 'before balancing')

    # Analyze the contribution of features in PCA 
    pca_contribution(pca, Features)

    # Balance the dataset to ensure equal distribution of stroke and non-stroke cases
    balanceHealthCareDataFrame = balance_data(healthCareDataFrame)

    # Split the balanced dataset into training and testing sets
    X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = split_data(balanceHealthCareDataFrame, 'stroke')

    # Perform PCA again on the balanced dataset
    X_pca_balanced, pca_balanced = pca_analysis(X_train_balanced)

    # Plot the scree plot after balancing to see the effect of balancing on feature importance 
    plot_scree_plot(pca_balanced, 'after balancing')

    # Analyze the contribution of features in PCA after balancing 
    pca_contribution(pca_balanced, Features)

    logger.info("Feature analysis completed.")
    return X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced, X_pca_balanced, pca_balanced


def train_models(X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced, X_pca_balanced, pca_balanced):
    """
    Train models on full features and PCA-reduced features, then save results.
    
    """
    logger.info("Starting model training...")

    # Train models using all features and perform grid search for hyperparameter tuning
    ModelsResultsAllFeatures = grid_search_func(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, Models, ParametersForGridSearch)

    # Select top 2 principal components and train models with reduced feature set
    X_train_pca_2, X_test_pca_2 = select_pca_components(X_pca_balanced, X_test_balanced, pca_balanced, 2)
    ModelsResults2PCA = grid_search_func(X_train_pca_2, y_train_balanced, X_test_pca_2, y_test_balanced, Models, ParametersForGridSearch)

    # Select top 8 principal components and train models reduced feature set
    X_train_pca_8, X_test_pca_8 = select_pca_components(X_pca_balanced, X_test_balanced, pca_balanced, 8)
    ModelsResults8PCA = grid_search_func(X_train_pca_8, y_train_balanced, X_test_pca_8, y_test_balanced, Models, ParametersForGridSearch)


    # Save results to CSV files
    ModelsResultsAllFeatures.to_csv("ModelsResults_AllFeatures.csv", index=False)
    ModelsResults2PCA.to_csv("ModelsResults_2PCA.csv", index=False)
    ModelsResults8PCA.to_csv("ModelsResults_8PCA.csv", index=False)

    logger.info("Model results saved successfully as CSV files.")
