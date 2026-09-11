"""
Cross-check model.js against the real sklearn bundle it was exported from.

Run from perio-screening-tool/:
    python3 test/verify_against_python.py

Requires the Questionnaire_validation repo's venv (joblib, scikit-learn,
pandas) and its models/ directory to be reachable at BUNDLE_PATH below.
"""
import glob
import json
import os
import subprocess
import sys

import joblib
import pandas as pd

_RESEARCH_MODELS_DIR = (
    "/Users/elena/Library/CloudStorage/OneDrive/Perio_screening/"
    "Questionnaire_validation/models")


def _latest_lr_bundle(models_dir=_RESEARCH_MODELS_DIR):
    """
    Newest perio_binary_LR_* bundle directory, or None.

    A hardcoded date here goes stale the moment a fresh bundle is trained —
    which is exactly what happened to this file once already (it named
    perio_binary_LR_2026-08-07 long after that bundle stopped being current).
    Sorting by name works because the date is embedded YYYY-MM-DD and an
    optional cohort suffix sorts after it, so lexicographic order is
    chronological order here.
    """
    candidates = sorted(glob.glob(os.path.join(models_dir, 'perio_binary_LR_*')))
    return candidates[-1] if candidates else None


BUNDLE_PATH = os.environ.get('QV_BUNDLE_PATH') or _latest_lr_bundle()

# The 'compact' engineered arm: symptom_count replaces the five yes/no symptom
# items, age is linear (no age_over_39 / age_spline term — both were tried and
# neither beat the linear term in nested CV; see benchmark.py's FEATURE_SETS).
# If the shipped arm changes again, this list and to_model_row() below are the
# two places that have to follow it — there is no bundle-driven auto-detection
# here on purpose, so that a silent arm change cannot make this test silently
# pass by comparing the wrong columns; it will instead fail loudly against
# whatever bundle BUNDLE_PATH resolves to, which is the point of the test.
FEATURES = ['symptom_count', 'Age', 'Sex', 'Smoking', 'Q2_own_teeth_gum_health']

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
    dict(Age=18, Sex='male', Smoking=False,
         Q1_gum_disease=False, Q3_gum_treatment=False, Q4_loose_teeth=True,
         Q5_bone_loss=False, Q6_tooth_appearance=False,
         Q2_own_teeth_gum_health=4),
    dict(Age=90, Sex='female', Smoking=True,
         Q1_gum_disease=True, Q3_gum_treatment=True, Q4_loose_teeth=False,
         Q5_bone_loss=True, Q6_tooth_appearance=True,
         Q2_own_teeth_gum_health=5),
]

SYMPTOM_KEYS = ['Q1_gum_disease', 'Q3_gum_treatment', 'Q4_loose_teeth',
                'Q5_bone_loss', 'Q6_tooth_appearance']


def to_model_row(patient):
    symptom_count = sum(bool(patient[k]) for k in SYMPTOM_KEYS)
    return {
        'symptom_count': symptom_count,
        'Age': patient['Age'],
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
    if not BUNDLE_PATH:
        raise SystemExit(
            f"no perio_binary_LR_* bundle found under {_RESEARCH_MODELS_DIR}. "
            f"Run src/run_binary_experiments.py in the research repo first, or "
            f"set QV_BUNDLE_PATH explicitly.")
    print(f"Bundle: {BUNDLE_PATH}")
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
