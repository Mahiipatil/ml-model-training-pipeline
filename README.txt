
# DS Assignment Submission

This package contains the solution for the Data Science internship screening assignment.

## Contents

- `iris.csv`:
  The dataset file used for training and testing.

- `algoparams_from_ui.json.rtf`:
  JSON configuration file (in RTF format) containing target column, selected features, imputation strategy, feature reduction method, and model settings.

- `solution.py`:
  A fully functional Python script that:
    - Reads the dataset and configuration.
    - Handles missing values.
    - Applies specified feature reduction.
    - Trains the specified model using GridSearchCV.
    - Prints the best parameters and score.

## How to Run

1. Ensure you have Python installed (Python 3.7+ recommended).
2. Install dependencies (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
   Required packages: `pandas`, `numpy`, `scikit-learn`.

3. Run the script:
   ```bash
   python solution.py
   ```

## Output

The script prints the best hyperparameters and best score achieved by GridSearchCV.

---
Good luck!
