import pandas as pd
from sklearn.utils import shuffle
from logger import logger 

def balance_data(df, target_col='stroke'):
    """
    Balances the dataset by randomly selecting an equal number of non-stroke cases 
    as there are stroke cases.

    """
    try:
        logger.info("Starting the balancing process...")

        # Count the number of samples before balancing
        before_balancing = df[target_col].value_counts()
        logger.info(f"Before balancing:\n{before_balancing}")

        # Separate the dataset into stroke and non-stroke cases
        stroke_cases = df[df[target_col] == 1]
        non_stroke_cases = df[df[target_col] == 0]

        # Randomly select an equal number of non-stroke cases as there are stroke cases
        random_non_stroke = non_stroke_cases.sample(n=len(stroke_cases), random_state=42)

        # Combine the stroke and selected non-stroke cases
        balanced_df = pd.concat([stroke_cases, random_non_stroke])

        # Shuffle the balanced dataset
        balanced_df = shuffle(balanced_df, random_state=42).reset_index(drop=True)

        # Count the number of samples after balancing
        after_balancing = balanced_df[target_col].value_counts()
        logger.info(f"\nAfter balancing:\n{after_balancing}")

        logger.info("Balancing completed successfully.")

        return balanced_df

    except Exception as e:
        logger.error(f"Error during data balancing: {e}")
        return None
