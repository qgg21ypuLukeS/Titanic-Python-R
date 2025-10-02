#####Library imports#####
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


#####Basic exploration of the data#####

titanic = pd.read_csv('data/titanic_train.csv')

list(titanic)

#print(titanic.head()) #Prints the first 5 rows of the dataframe

#print(titanic.info()) #Prints a concise summary of the dataframe

#print(titanic.describe()) #Prints the summary statistics of the dataframe

#print(titanic.isnull().sum()) #Print number of missing values in each column

#print(len(titanic)) #Print length of the dataframe


#####Data Cleaning#####

#outlier detection 

titanic_filter = titanic.filter(['Age', 'Fare', 'Pclass'])

def boxplot_outliers(titanic_filter, figsize=(15, 10)):
    """
    Creates box plots for all columns in a dataframe, attempting each one.
    Skips columns that fail and reports which ones couldn't be plotted.
    """
    
    # Step 1: Get the total number of columns
    n_cols = len(titanic_filter.columns)
    
    # Step 2: Calculate grid dimensions (rows x columns)
    # We'll use 3 columns per row - you can adjust this!
    ncols = 3
    nrows = (n_cols + ncols - 1) // ncols  # This rounds up
    # Example: if n_cols=10, nrows = (10+3-1)//3 = 12//3 = 4
    
    # Step 3: Create the figure with all subplots at once
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Step 4: Flatten the axes array so we can iterate through it easily
    # axes might be 2D array like [[ax1, ax2, ax3], [ax4, ax5, ax6]]
    # flatten makes it 1D: [ax1, ax2, ax3, ax4, ax5, ax6]
    axes = axes.flatten()
    
    # Step 5: Loop through each column and each subplot together
    for idx, column in enumerate(titanic_filter.columns):
        try:
            # Try to create a box plot for this column
            sns.boxplot(data=titanic_filter, y=column, ax=axes[idx])
            axes[idx].set_title(f'{column}')  # Add column name as title
            
        except Exception as e:
            # If it fails, note which column failed
            print(f"Could not plot '{column}': {type(e).__name__}")
            axes[idx].set_visible(False)  # Hide the empty subplot
    
    # Step 6: Hide any extra unused subplots
    # If we have 10 columns but 12 subplots (4x3), hide the last 2
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    # Step 7: Adjust layout so plots don't overlap
    plt.tight_layout()
    plt.show()

boxplot_outliers(titanic_filter)  


#KNN imputer for missing values
#imputer = KNNImputer(n_neighbors=5)
#titanic_imputed 
#Decide if columns need to be dropped or filled with mean/median/mode

