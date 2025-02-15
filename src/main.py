from pipeline import preprocess_data, feature_analysis, train_models

def main():
    # Load the healthcare dataset into a pandas DataFrame 
    filepath = 'data/healthcare-dataset-stroke-data.csv'

    # Step 1: Preprocess the dataset (exploration, visualization, missing values handling, encoding)
    healthCareDataFrame = preprocess_data(filepath)

    # Step 2: Perform feature analysis (PCA, data balancing, scree plot, feature contributions)
    X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced, X_pca_balanced, pca_balanced = feature_analysis(healthCareDataFrame)

    # Step 3: Train models using all features and PCA-reduced features, then save results
    train_models(X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced, X_pca_balanced, pca_balanced)

if __name__ == '__main__':
    main()
