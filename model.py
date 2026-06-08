import pickle
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

MODEL_PATH         = os.path.join(MODELS_DIR, 'model.pkl')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.pkl')
METRICS_PATH       = os.path.join(MODELS_DIR, 'metrics.pkl')


def load_artifacts():
    """Load all pipeline artifacts from pkl files."""
    print("Loading pipeline artifacts...")

    with open(MODEL_PATH, 'rb') as f:
        ensemble_model = pickle.load(f)

    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)

    with open(METRICS_PATH, 'rb') as f:
        metrics = pickle.load(f)

    print("All artifacts loaded\n")
    return ensemble_model, feature_names, metrics


def preprocess_input(data: dict, feature_names: list) -> np.ndarray:
    """Transform raw passenger dictionary into processed numeric array."""
    df = pd.DataFrame([data])

    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
        "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare", 
        "Sir": "Rare", "Jonkheer": "Rare"
    }
    df['Title'] = df['Title'].replace(title_mapping)
    standard_titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Rare']
    df['Title'] = df['Title'].apply(lambda x: x if x in standard_titles else 'Rare')

    median_ages = {'Mr': 30.0, 'Mrs': 35.0, 'Miss': 21.0, 'Master': 4.0, 'Rare': 48.0}
    if pd.isna(df['Age'].iloc[0]) or df['Age'].iloc[0] is None:
        passenger_title = df['Title'].iloc[0]
        df['Age'] = median_ages.get(passenger_title, 29.0)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilyGroup'] = 'Small'
    df.loc[df['FamilySize'] == 1, 'FamilyGroup'] = 'Alone'
    df.loc[df['FamilySize'] > 4, 'FamilyGroup'] = 'Large'

    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'FamilyGroup'])

    df = df.reindex(columns=feature_names, fill_value=0)
    return df.to_numpy()


def predict_survival(input_data: dict) -> dict:
    """
    Predict passenger survival probability using the Voting Ensemble.

    Args:
        input_data: Dictionary of passenger features

    Returns:
        Dictionary with predicted outcome and ensemble details
    """
    ensemble_model, feature_names, metrics = load_artifacts()
    X = preprocess_input(input_data, feature_names)

    prediction = ensemble_model.predict(X)[0]
    probability = ensemble_model.predict_proba(X)[0][1]

    results = {
        'prediction': int(prediction),
        'survival_probability': round(float(probability), 4),
        'ensemble_metrics': {
            'mean_cv_accuracy': round(metrics['Voting_Ensemble']['Accuracy'], 4),
            'cv_std_dev': round(metrics['Voting_Ensemble']['Std_Dev'], 4)
        }
    }
    return results


def display_metrics():
    """Print a summary of model cross-validation metrics."""
    with open(METRICS_PATH, 'rb') as f:
        metrics = pickle.load(f)

    print(f"{'Model':<25} {'Mean Accuracy':>15} {'Std Dev':>10}")
    for model_name, m in metrics.items():
        name = model_name.replace('_', ' ').title()
        print(f"{name:<25} {m['Accuracy']:>15.4f} {m['Std_Dev']:>10.4f}")


if __name__ == '__main__':

    sample_input = {
        'Pclass': 3,
        'Name': 'Braund, Mr. Owen Harris',
        'Sex': 'male',
        'Age': None,
        'SibSp': 1,
        'Parch': 0,
        'Ticket': 'A/5 21171',
        'Fare': 7.25,
        'Cabin': None,
        'Embarked': 'S'
    }

    print("Titanic Passenger Survival Prediction Pipeline")
    print("\nInput Data:")
    for key, value in sample_input.items():
        print(f"  {key:<20} {value}")

    print("\nModel Performance Summary (Cross-Validation):")
    try:
        display_metrics()
    except Exception as e:
        print(f"  Could not load metrics file: {e}")

    print("\nRunning Inference Engine...")
    try:
        results = predict_survival(sample_input)
        status = "SURVIVED" if results['prediction'] == 1 else "DECEASED"
        print(f"\n  Predicted Outcome:       {status}")
        print(f"  Survival Probability:    {results['survival_probability'] * 100:.2f}%")
        print(f"  Ensemble CV Accuracy:    {results['ensemble_metrics']['mean_cv_accuracy']:.4f}")
    except Exception as e:
        print(f"  Inference error: {e}")