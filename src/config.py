from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# Columns related to categorical and numrical data
Categorical_Columns = ['gender', 'hypertension', 'heart_disease','ever_married','work_type','Residence_type', 'smoking_status', 'stroke']
Numeric_Columns = ['age','avg_glucose_level' ,'bmi']

# Encoding dictionary for categorical columns
Encoding_Dictionary = {
    'gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'ever_married': {'Yes': 1, 'No': 0},
    'work_type': {'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4},
    'Residence_type': {'Urban': 0, 'Rural': 1},
    'smoking_status': {'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2, 'smokes': 3}
}

Features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']


# Define the models
Models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Define parameter grids for each model
ParametersForGridSearch = {
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd']
    },
    'SVM': {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
}
}