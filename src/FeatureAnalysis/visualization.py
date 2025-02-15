import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from logger import logger

# Define a directory to save the plots
output_dir = "plots" 

# Make sure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def plot_correlation_heatmap(df, method='pearson'):
    """
    Calculates and plots a heatmap for the correlation matrix of the given DataFrame.

    """
    logger.info(f"Calculating {method} correlation matrix...")

    # Compute correlation matrix
    try:
        corr_mat = df.corr(method=method)
        logger.info("Correlation matrix calculated successfully.")
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {e}")
        return

    # Plot heatmap
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Correlation Heatmap ({method.capitalize()} Method)')
        
        # Save the heatmap plot
        plot_filename = os.path.join(output_dir, f"correlation_heatmap_{method}.png")
        plt.savefig(plot_filename)
        plt.close() 

        logger.info(f"Correlation heatmap saved as {plot_filename}.")
    except Exception as e:
        logger.error(f"Error plotting heatmap: {e}")


def plot_scree_plot(pca, status):
    """
    Plots the Scree plot showing the explained variance ratio of each principal component.
    
    """
    try:
        logger.info(f"Plotting the Scree plot for status: {status}...")

        # Get the explained variance ratio
        explained_variance = pca.explained_variance_ratio_

        # Plotting the Scree plot
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
        plt.title(f'Scree Plot - {status}')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.grid(True)

        # Save the Scree plot
        plot_filename = os.path.join(output_dir, f"scree_plot_{status}.png")
        plt.savefig(plot_filename)
        plt.close()

        logger.info(f"Scree plot saved as {plot_filename}.")

        # Log the explained variance percentages
        explained_variance_percent = explained_variance * 100
        logger.info("Explained variance by each principal component:")
        for i, var in enumerate(explained_variance_percent, 1):
            logger.info(f"PC{i}: {var:.2f}%")

    except Exception as e:
        logger.error(f"Error in plotting Scree plot: {e}")