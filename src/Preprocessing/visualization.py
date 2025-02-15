import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from logger import logger 

# Define a directory to save the plots
output_dir = "plots" 

# Make sure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def plot_categorical_data(df, categorical_columns):
    """
    Function to visualize the distribution of categorical columns in the dataset.

    """
    logger.info("Starting categorical data visualization...")
    
    # Iterate through the list of categorical columns and plot bar charts
    for column in categorical_columns:
        if column in df.columns:
            value_counts = df[column].value_counts()
            logger.info(f"Value counts for {column}:\n{value_counts}")

            # Plot bar chart
            plt.figure(figsize=(6, 4))
            value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title(f"Value Counts for {column}")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.tight_layout()

            # Save the plots
            plot_filename = os.path.join(output_dir, f"{column}_value_counts.png")
            plt.savefig(plot_filename)
            plt.close()
            
            logger.info(f"Bar chart saved as {plot_filename}.")
        else:
            logger.warning(f"Column {column} not found in DataFrame.")

    logger.info("Categorical data visualization completed.")

def plot_numerical_data(df, numeric_columns):
    """
    Function to visualize the distribution of numerical columns in the dataset.

    """
    logger.info("Starting numerical data visualization...")

    # Iterate through the list of numeric columns and plot histograms 
    for column in numeric_columns:
        if column in df.columns:
            plt.figure(figsize=(6, 4))
            df[column].plot(kind='hist', bins=10, color='lightgreen', edgecolor='black')
            plt.title(f"Histogram for {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.tight_layout()

            # Save the plots
            plot_filename = os.path.join(output_dir, f"{column}_histogram.png")
            plt.savefig(plot_filename)
            plt.close()  
            
            logger.info(f"Histogram saved as {plot_filename}.")
        else:
            logger.warning(f"Column {column} not found in DataFrame.")

    logger.info("Numerical data visualization completed.")
