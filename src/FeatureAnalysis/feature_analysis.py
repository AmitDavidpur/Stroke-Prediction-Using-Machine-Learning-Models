from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logger import logger
from sklearn.decomposition import PCA
import pandas as pd

def split_data(df, outcome):
    """    
    Splits the data into training and testing sets, and standardizes the features.

    """
    try:
        logger.info("Starting the data splitting process...")
        
        # Check if outcome column exists in the DataFrame
        if outcome not in df.columns:
            logger.error(f"Outcome column '{outcome}' not found in DataFrame.")
            return None, None, None, None
        
        # Split the data into features and outcome
        y = df[outcome].astype(int)
        X = df.drop(columns=[outcome])
        
        logger.info("Data successfully split into features and outcome.")
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info("Features successfully standardized.")
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        logger.info("Data successfully split into training and testing sets.")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error in split_data function: {e}")
        return None, None, None, None

def pca_analysis(X_train):
    """
    Perform PCA analysis on the provided training data.

    """
    try:
        logger.info("Starting PCA analysis...")

        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_train)

        logger.info("PCA analysis completed successfully.")
        
        return X_pca, pca
    except Exception as e:
        logger.error(f"Error in PCA analysis: {e}")
        return None, None


def pca_contribution(pca, features):
    """
    Evaluates the contribution of the features to the first two principal components (PC1 and PC2).
    
    """
    try:
        logger.info("Evaluating the contribution of features to PCA dimensions...")
        
        # Get the loadings for PC1 and PC2
        loadings = pca.components_.T[:, :2]

        # Create a DataFrame to display the loadings
        loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=features)

        # Get absolute values of the loadings
        abs_loadings_df = round(loadings_df.abs(), 2)

        # Sort the features by their absolute contribution for PC1
        sorted_pc1 = abs_loadings_df['PC1'].sort_values(ascending=False)

        # Sort the features by their absolute contribution for PC2
        sorted_pc2 = abs_loadings_df['PC2'].sort_values(ascending=False)

        # Log the sorted contributions
        logger.info("Sorted Contributions of Features to PC1:")
        logger.info(sorted_pc1)
        logger.info("\nSorted Contributions of Features to PC2:")
        logger.info(sorted_pc2)

    except Exception as e:
        logger.error(f"Error in evaluating PCA contributions: {e}")
