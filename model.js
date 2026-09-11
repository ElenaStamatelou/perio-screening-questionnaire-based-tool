/*
 * Periodontitis screening model — logistic regression, 5 features.
 *
 * Exported from perio_binary_LR_2026-09-08_nijland/ in the
 * Questionnaire_validation research repo (model.pkl + scaler.pkl +
 * metadata.json), full precision, by src/export_model_js.py.
 * Training cohorts: 0N, 1N (n=341, 56 events).
 * Selected by the calibrated_auc rule (calibration slope in [0.8, 1.2], highest AUC among eligible models): AUC 0.775, slope 0.97, O:E 0.99 on pooled nested CV, cohorts 0N+1N.
 * Do not hand-edit these numbers — re-export from the bundle if the shipped model changes.
 */

const MODEL = {
  intercept: -4.625756792259177,
  coefficients: {
    symptom_count: 3.1668954601639645,
    Age: 2.1572321741392084,
    Sex: 0.17401791569480946,
    Smoking: 1.035073870702638,
    Q2_own_teeth_gum_health: 1.1613749890572775,
  },
  // MinMaxScaler bounds from training data, applied before the dot product.
  scaler: {
    symptom_count: { min: 0.0, max: 5.0 },
    Age: { min: 18.0, max: 90.0 },
    Sex: { min: 0.0, max: 1.0 },
    Smoking: { min: 0.0, max: 1.0 },
    Q2_own_teeth_gum_health: { min: 1.0, max: 5.0 },
  },
};

// The five yes/no symptom questions summed into `symptom_count`.
const SYMPTOM_KEYS = ['Q1_gum_disease', 'Q3_gum_treatment', 'Q4_loose_teeth',
                       'Q5_bone_loss', 'Q6_tooth_appearance'];

/**
 * answers: { Q1_gum_disease (bool), Q3_gum_treatment (bool), Q4_loose_teeth (bool), Q5_bone_loss (bool), Q6_tooth_appearance (bool), Age, Sex ('male'|'female'), Smoking (bool), Q2_own_teeth_gum_health }
 *
 * Returns P(severe periodontitis), a number in [0, 1].
 */
function predictSevereRisk(answers) {
  const symptom_count = SYMPTOM_KEYS.reduce(
    (sum, key) => sum + (answers[key] ? 1 : 0), 0);
  const raw = {
    symptom_count,
    Age: answers.Age,
    Sex: answers.Sex === 'male' ? 1 : 0,
    Smoking: answers.Smoking ? 1 : 0,
    Q2_own_teeth_gum_health: answers.Q2_own_teeth_gum_health,
  };

  let logit = MODEL.intercept;
  for (const key in MODEL.coefficients) {
    const { min, max } = MODEL.scaler[key];
    const scaled = (raw[key] - min) / (max - min);
    logit += MODEL.coefficients[key] * scaled;
  }

  return 1 / (1 + Math.exp(-logit));
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { predictSevereRisk, MODEL, SYMPTOM_KEYS };
}
