/*
 * Periodontitis screening model — logistic regression, 6 engineered features.
 *
 * Exported from models/perio_binary_LR_2026-08-07/ in the Questionnaire_validation
 * research repo (model.pkl + scaler.pkl + metadata.json), full precision, on 2026-08-10.
 * Internal pooled nested-CV performance at export time: AUC 0.785, calibration
 * slope 0.94, O:E 0.99 (n=469 patients, 3 Dutch cohorts). Do not hand-edit these
 * numbers — re-export from the bundle if the shipped model changes.
 */

const MODEL = {
  intercept: -4.651092962141181,
  coefficients: {
    symptom_count: 2.1594217563165663,
    Age: 0.7746785472750496,
    age_over_39: 0.9113153681722603,
    Sex: 0.43154200160525213,
    Smoking: 0.6512138194854925,
    Q2_own_teeth_gum_health: 1.7514250837943097,
  },
  // MinMaxScaler bounds from training data, applied before the dot product.
  scaler: {
    symptom_count: { min: 0, max: 5 },
    Age: { min: 18, max: 90 },
    age_over_39: { min: 0, max: 1 },
    Sex: { min: 0, max: 1 },
    Smoking: { min: 0, max: 1 },
    Q2_own_teeth_gum_health: { min: 1, max: 5 },
  },
};

// The five yes/no symptom questions summed into `symptom_count`.
const SYMPTOM_KEYS = ['Q1_gum_disease', 'Q3_gum_treatment', 'Q4_loose_teeth',
                       'Q5_bone_loss', 'Q6_tooth_appearance'];

/**
 * answers: { Age, Sex ('male'|'female'), Smoking (bool),
 *            Q1_gum_disease, Q3_gum_treatment, Q4_loose_teeth, Q5_bone_loss,
 *            Q6_tooth_appearance (all bool), Q2_own_teeth_gum_health (1-5) }
 *
 * Returns P(severe periodontitis), a number in [0, 1].
 */
function predictSevereRisk(answers) {
  const symptom_count = SYMPTOM_KEYS.reduce(
    (sum, key) => sum + (answers[key] ? 1 : 0), 0);

  const raw = {
    symptom_count,
    Age: answers.Age,
    age_over_39: answers.Age > 39 ? 1 : 0,
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
