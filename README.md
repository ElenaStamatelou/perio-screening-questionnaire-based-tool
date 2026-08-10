# Gum disease risk screening — prototype

A single static page: nine questions in, an estimated risk of severe gum
disease (periodontitis) out. No backend — the logistic regression model runs
entirely in the browser (`model.js`), so nothing typed into the form is sent
or stored anywhere.

**Status: prototype / demo.** This is not a validated clinical tool. The
model was trained on Dutch cohort data using the validated Dutch wording of
Verhulst et al. (2019); the English questions on this page are an
unvalidated adaptation for demonstration purposes. See the "About this tool"
panel on the result screen for the full caveats, and
`docs/web_tool_design.md` in the source research repo
(`Questionnaire_validation`, private) for the underlying design decisions.

## Run locally

No build step. Either open `index.html` directly in a browser, or serve it:

```
python3 -m http.server 8000
```

then visit `http://localhost:8000`.

## Correctness check

`model.js` is exported from the sklearn model bundle
(`models/perio_binary_LR_2026-08-07/`) in the source research repo, at full
floating-point precision. `test/verify_against_python.py` re-loads that
bundle with joblib and checks that `model.js`'s predictions agree with
`model.predict_proba()` to within 1e-9, for five fixed test patients. Run it
whenever `model.js` is regenerated:

```
python3 test/verify_against_python.py
```

(Requires the research repo's venv — joblib, scikit-learn, pandas — and
access to the model bundle.)

## Deploying to GitHub Pages

1. Push this directory to a public GitHub repo.
2. Repo Settings → Pages → Deploy from branch → `main`, root.
3. The page is served at `https://<user>.github.io/<repo>/`.

No GitHub Actions or build step needed — it's plain HTML/CSS/JS.
