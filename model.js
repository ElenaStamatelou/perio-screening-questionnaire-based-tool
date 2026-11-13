/*
 * Periodontitis screening model — logistic regression, 5 features.
 *
 * Exported from models\perio_binary_LR_2026-11-03_nijland/ in the
 * Questionnaire_validation research repo (model.pkl + scaler.pkl +
 * metadata.json), full precision, by src/export_model_js.py.
 * Training cohorts: 0N, 1N (n=341, 56 events).
 * Selected by calibrated_auc on pooled nested CV (cohorts ['0N', '1N'], severe_vs_nonsevere, engineering=yes): AUC 0.779, slope 0.92, O:E 1.00.
 * Do not hand-edit these numbers — re-export from the bundle if the shipped model changes.
 */

const MODEL = {
  intercept: -4.625756792259178,
  coefficients: {
    symptom_count: 3.166895460163963,
    Age: 2.1572321741392093,
    Sex: 0.1740179156948092,
    Smoking: 1.035073870702639,
    Q2_own_teeth_gum_health: 1.161374989057279,
  },
  // MinMaxScaler bounds from training data, applied before the dot product.
  scaler: {
    symptom_count: { min: 0.0, max: 5.0 },
    Age: { min: 18.0, max: 90.0 },
    Sex: { min: 0.0, max: 1.0 },
    Smoking: { min: 0.0, max: 1.0 },
    Q2_own_teeth_gum_health: { min: 1.0, max: 5.0 },
  },
  // Referral cut-off on the predicted probability (metadata.json operating_threshold).
  operatingThreshold: 0.17,
  // Mean SCALED value of each feature over the 341 training patients: the reference of
  // explainSevereRisk() (SHAP values with the training data as background).
  trainingMeans: {
    symptom_count: 0.1906158357771261,
    Age: 0.5163734115347018,
    Sex: 0.5395894428152492,
    Smoking: 0.15542521994134897,
    Q2_own_teeth_gum_health: 0.5439882697947214,
  },
};

// The five yes/no symptom questions summed into `symptom_count`.
const SYMPTOM_KEYS = ['Q1_gum_disease', 'Q3_gum_treatment', 'Q4_loose_teeth',
                       'Q5_bone_loss', 'Q6_tooth_appearance'];

/**
 * answers: { Q1_gum_disease (bool), Q3_gum_treatment (bool), Q4_loose_teeth (bool), Q5_bone_loss (bool), Q6_tooth_appearance (bool), Age, Sex ('male'|'female'), Smoking (bool), Q2_own_teeth_gum_health }
 *
 * Returns the model's inputs before scaling, keyed by feature.
 */
function modelInputs(answers) {
  const symptom_count = SYMPTOM_KEYS.reduce(
    (sum, key) => sum + (answers[key] ? 1 : 0), 0);
  return {
    symptom_count,
    Age: answers.Age,
    Sex: answers.Sex === 'male' ? 1 : 0,
    Smoking: answers.Smoking ? 1 : 0,
    Q2_own_teeth_gum_health: answers.Q2_own_teeth_gum_health,
  };
}

function scaledInput(raw, key) {
  const { min, max } = MODEL.scaler[key];
  return (raw[key] - min) / (max - min);
}

/**
 * Returns P(severe periodontitis), a number in [0, 1].
 */
function predictSevereRisk(answers) {
  const raw = modelInputs(answers);
  let logit = MODEL.intercept;
  for (const key in MODEL.coefficients) {
    logit += MODEL.coefficients[key] * scaledInput(raw, key);
  }
  return 1 / (1 + Math.exp(-logit));
}

/**
 * Why this result: the SHAP value of each feature, on the log-odds scale, relative to the average
 * training patient. For this logistic regression they are exact (coefficient x (scaled value - training
 * mean)); positive raises the predicted probability, negative lowers it. They add up to
 * logit(prediction) - logit(average patient). Returns { values: {feature: shap}, inputs: raw inputs }.
 */
function explainSevereRisk(answers) {
  const raw = modelInputs(answers);
  const values = {};
  for (const key in MODEL.coefficients) {
    values[key] = MODEL.coefficients[key] * (scaledInput(raw, key) - MODEL.trainingMeans[key]);
  }
  return { values, inputs: raw };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { predictSevereRisk, MODEL, SYMPTOM_KEYS, explainSevereRisk };
}
