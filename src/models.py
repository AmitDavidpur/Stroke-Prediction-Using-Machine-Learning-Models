from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from logger import logger 
from sklearn.model_selection import GridSearchCV
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """
    Function to evaluate model performance using various metrics.

    """
    try:
        logger.info("Starting model performance evaluation...")

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f_score = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        miss_rate = fn / (fn + tp)
        fallout_rate = fp / (fp + tn)

        # Log the metrics
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f_score:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Miss Rate: {miss_rate:.4f}")
        logger.info(f"Fallout Rate: {fallout_rate:.4f}")

        logger.info("Model performance evaluation completed successfully.")

        return precision, recall, f_score, accuracy, miss_rate, fallout_rate

    except Exception as e:
        logger.error(f"Error during model performance evaluation: {e}")
        return None




def grid_search_func(X_train, y_train, X_test, y_test, models, param_grids):
    """
    Function to perform grid search on multiple models and evaluate them using different metrics.
    """
    # Initialize an empty list to store results
    results_list = []

    try:
        logger.info("Starting grid search for model selection...")

        # Loop through each model and perform grid search
        for model_name in models:
            logger.info(f"Performing grid search for {model_name}...")

            # Perform Grid Search
            grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Get the best model and make predictions
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # Get best parameters for the model
            best_params = grid_search.best_params_
            logger.info(f"{model_name}'s best parameters: {best_params}")

            # Calculate the metrics for the best model
            precision, recall, f_score, accuracy, miss_rate, fallout_rate = calculate_metrics(y_test, y_pred)

            # Store the results in the results list
            results = {
                'Model': model_name,
                'Precision': round(precision, 2),
                'Recall': round(recall, 2),
                'F-Score': round(f_score, 2),
                'Accuracy': round(accuracy, 2),
                'Miss rate': round(miss_rate, 2),
                'Fall-out rate': round(fallout_rate, 2),
            }

            # Append the results to the list
            results_list.append(results)

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results_list)
        logger.info("Grid search completed successfully.")

        return results_df

    except Exception as e:
        logger.error(f"An error occurred during grid search: {e}")
        return None
    

def select_pca_components(X_train, X_test, pca_model, n_components):
    """
    Selects the first n principal components from the training and test sets.
    """
    try:
        # Select the first n components from the training set
        X_train_selected = X_train[:, :n_components]

        # Apply the PCA transformation on the test set
        X_test_selected = pca_model.transform(X_test)[:, :n_components]

        logger.info(f"Successfully selected the first {n_components} principal components.")
        return X_train_selected, X_test_selected

    except Exception as e:
        logger.error(f"Error selecting PCA components: {e}")
        return None, None

