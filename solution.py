import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
iris_data = pd.read_csv('/content/drive/MyDrive/iris.csv')

# Step 2: Parse the model parameters from JSON embedded in RTF
with open('/content/algoparams_from_ui.json.rtf', 'r') as rtf_file:
    rtf_content = rtf_file.read()
    json_start = rtf_content.find('{')
    json_end = rtf_content.rfind('}') + 1
    json_string = rtf_content[json_start:json_end]
    model_config = json.loads(json_string)

# Step 3: Extract configuration values
target_col = model_config.get('target')
problem_type = model_config.get('type')  # 'classification' or 'regression'
model_name = model_config.get('model')
model_params = model_config.get('params', {})

# Step 4: Data Preprocessing
features = iris_data.drop(columns=[target_col])
labels = iris_data[target_col]

if labels.dtype == 'object':
    labels = LabelEncoder().fit_transform(labels)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Step 5: Model Selection
model_mapping = {
    'RandomForestClassifier': RandomForestClassifier,
    'RandomForestRegressor': RandomForestRegressor,
    'LogisticRegression': LogisticRegression,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'SVC': SVC
}

selected_model_cls = model_mapping.get(model_name)

if selected_model_cls is None:
    raise ValueError(f"Unsupported model: {model_name}")

# Step 6: Build and Optimize Pipeline
pipeline_steps = [
    ('pca', PCA(n_components=min(3, features.shape[1]))),
    ('model', selected_model_cls(**model_params))
]

pipeline = Pipeline(pipeline_steps)

# Step 7: Grid Search (optional)
if 'grid_params' in model_config:
    search = GridSearchCV(pipeline, model_config['grid_params'], cv=5)
    search.fit(features_imputed, labels)
    best_model = search.best_estimator_
else:
    pipeline.fit(features_imputed, labels)
    best_model = pipeline

# Step 8: Save the model (example - can be modified)
import joblib
joblib.dump(best_model, 'trained_model.pkl')

print("Model training completed and saved as 'trained_model.pkl'.")
