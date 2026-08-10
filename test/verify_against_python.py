"""
Cross-check model.js against the real sklearn bundle it was exported from.

Run from perio-screening-tool/:
    python3 test/verify_against_python.py

Requires the Questionnaire_validation repo's venv (joblib, scikit-learn,
pandas) and its models/ directory to be reachable at BUNDLE_PATH below.
"""
import json
import subprocess
import sys

import joblib
import pandas as pd

BUNDLE_PATH = ("/Users/elena/Library/CloudStorage/OneDrive/"
               "Questionnaire_validation/models/perio_binary_LR_2026-08-07")

FEATURES = ['symptom_count', 'Age', 'age_over_39', 'Sex', 'Smoking',
            'Q2_own_teeth_gum_health']

# Fixed test patients, in the JS-facing answer shape.
PATIENTS = [
    dict(Age=25, Sex='female', Smoking=False,
         Q1_gum_disease=False, Q3_gum_treatment=False, Q4_loose_teeth=False,
         Q5_bone_loss=False, Q6_tooth_appearance=False,
         Q2_own_teeth_gum_health=1),
    dict(Age=70, Sex='male', Smoking=True,
         Q1_gum_disease=True, Q3_gum_treatment=True, Q4_loose_teeth=True,
         Q5_bone_loss=True, Q6_tooth_appearance=True,
         Q2_own_teeth_gum_health=5),
    dict(Age=39, Sex='male', Smoking=False,
         Q1_gum_disease=True, Q3_gum_treatment=False, Q4_loose_teeth=False,
         Q5_bone_loss=True, Q6_tooth_appearance=False,
         Q2_own_teeth_gum_health=3),
    dict(Age=40, Sex='female', Smoking=True,
         Q1_gum_disease=False, Q3_gum_treatment=True, Q4_loose_teeth=False,
         Q5_bone_loss=False, Q6_tooth_appearance=True,
         Q2_own_teeth_gum_health=2),
    dict(Age=55, Sex='male', Smoking=False,
         Q1_gum_disease=False, Q3_gum_treatment=False, Q4_loose_teeth=True,
         Q5_bone_loss=False, Q6_tooth_appearance=False,
         Q2_own_teeth_gum_health=4),
]

SYMPTOM_KEYS = ['Q1_gum_disease', 'Q3_gum_treatment', 'Q4_loose_teeth',
                'Q5_bone_loss', 'Q6_tooth_appearance']


def to_model_row(patient):
    symptom_count = sum(bool(patient[k]) for k in SYMPTOM_KEYS)
    return {
        'symptom_count': symptom_count,
        'Age': patient['Age'],
        'age_over_39': 1.0 if patient['Age'] > 39 else 0.0,
        'Sex': 1.0 if patient['Sex'] == 'male' else 0.0,
        'Smoking': 1.0 if patient['Smoking'] else 0.0,
        'Q2_own_teeth_gum_health': patient['Q2_own_teeth_gum_health'],
    }


def python_predictions():
    model = joblib.load(f'{BUNDLE_PATH}/model.pkl')
    scaler = joblib.load(f'{BUNDLE_PATH}/scaler.pkl')
    rows = [to_model_row(p) for p in PATIENTS]
    X = pd.DataFrame(rows, columns=FEATURES)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1].tolist()


def js_predictions():
    script = f"""
    const {{ predictSevereRisk }} = require('{__file__.rsplit('/', 1)[0]}/../model.js');
    const patients = {json.dumps(PATIENTS)};
    const out = patients.map(predictSevereRisk);
    console.log(JSON.stringify(out));
    """
    result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError('node script failed')
    return json.loads(result.stdout)


def main():
    py = python_predictions()
    js = js_predictions()
    print(f"{'patient':>8}  {'python':>12}  {'js':>12}  {'abs diff':>12}")
    max_diff = 0.0
    for i, (p, j) in enumerate(zip(py, js)):
        diff = abs(p - j)
        max_diff = max(max_diff, diff)
        print(f"{i:>8}  {p:12.10f}  {j:12.10f}  {diff:12.2e}")
    print(f"\nmax abs diff: {max_diff:.2e}")
    assert max_diff < 1e-9, "JS and Python predictions disagree beyond 1e-9"
    print("OK — model.js matches the sklearn bundle to within 1e-9.")


if __name__ == '__main__':
    main()
