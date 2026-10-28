# Loan Default Prediction: Balancing Approval and Risk

**Course:** INSY 674  
**Status:** Data split, EDA, preprocessing pipeline and baseline models completed; model comparison, fine-tuning and final evaluation in progress.  
**Dataset:** [Lending Club loan dataset for granting models](https://zenodo.org/records/11295916)

## 1. Project Overview

A lender wants to approve borrowers who can repay while keeping defaults under control. Approving more loans can increase lending volume, but it may also increase losses. Rejecting too many applications can turn away borrowers who would have repaid.

Our main question is:

> If we approve the lowest-risk loans, how does the default rate change as we approve more or fewer loans?

We use information available at the application stage to predict default risk. We then compare models and explore the trade-off between approval rate and default rate among approved loans.

This is a class project using historical issued loans. Its results do not establish that the model is ready for real lending decisions.

## 2. Business Objective and Problem Framing

### Business objective

Identify lower-risk loans while retaining a meaningful share of lending volume.

We will compare models at similar approval rates to see whether additional features and more complex models improve on a simple baseline.

We do not have a specific business risk target or a complete financial cost model. Therefore, we evaluate the approval–default trade-off rather than claiming to find the most profitable policy.

### Who would use the model?

- **Credit-risk analysts:** compare borrower risk and investigate model errors.
- **Lending-policy teams:** evaluate different approval thresholds.

A possible workflow is:

1. Collect application information.
2. Apply the fitted preprocessing pipeline.
3. Generate a default-risk score.
4. Apply a threshold selected using validation data.
5. Review performance and outcomes over time.

### Machine learning task

This is a **supervised binary classification** problem.

- **Unit of analysis:** one loan.
- **Target:** `Default`.
- **Positive class, 1:** charged off.
- **Negative class, 0:** fully paid.
- **Training:** offline, using historical loans.
- **Output:** a score where higher values indicate higher default risk.

If we interpret scores as default probabilities, we also need to assess calibration.

### Baseline

Manual review and credit-score rules are common lending approaches. Our baseline is **FICO-only logistic regression**, which helps us test whether additional features improve predictions.

This baseline does not reproduce Lending Club’s actual historical decision process.

## 3. Team Roles and Responsibilities

Each person documents their own work and presents for about two minutes. The project lead combines the sections into a consistent story.

| Member | Student ID | Role | Main responsibilities | Cross-reviewer |
|---|---|---|---|---|
| Dora Zhuang | 261012351 | Project Lead & Storyteller | Business framing, metrics, README, assumptions, limitations, slides and presentation flow | TBD |
| TBD | TBD | Data Exploration | Variable profiles, data quirks, charts and takeaways | TBD |
| TBD | TBD | Data Preparation & Pipeline / Repo Owner | Time split, sampling, shared pipeline, repo structure and merges | TBD |
| TBD | TBD | Baseline & Shortlist | Baseline, model comparison, feature importance and error analysis | TBD |
| TBD | TBD | Fine-tuning & Final Evaluation | Hyperparameter search, ensemble, threshold selection and final test evaluation | TBD |

### Shared rules

- Use the same data split and shared preprocessing pipeline.
- Fit learned preprocessing inside each training fold.
- Keep test data out of EDA, feature selection, tuning, and threshold selection.
- Assign a cross-reviewer to each part, especially the pipeline.
- Push work regularly so teammates are not blocked.
- Keep detailed methods in notebooks and summarize the main findings here.
- Replace `TBD` with actual results before submission.
- Clearly distinguish planned work from completed experiments.

## 4. Data Source and Split

### Data source

The [source dataset](https://doi.org/10.5281/zenodo.11295916) contains:

- **1,347,681 loans**
- **15 columns**
- Loans issued between **2007 and 2018**
- Application-time explanatory variables
- Final outcomes of fully paid or charged off

The dataset creators excluded loans in transitional states. Therefore, the data do not include every issued loan from each year.

The dataset also contains only issued loans. We do not observe applicants who were rejected.

### Completed train/test split

The splitting notebook separates origination periods before creating stratified samples.

| Dataset | Issue years | Available pool | Sample size | Default rate |
|---|---|---:|---:|---:|
| Training / development | 2007–2015 | 829,347 | 150,000 | 18.46% |
| Reserved test | 2016 | 293,057 | 50,000 | 23.28% |

- Sampling is stratified by `Default`.
- Random seed: **42**.
- Sampled training and test IDs are checked for overlap.
- EDA uses the training sample.

Sampling keeps model comparisons manageable within our project timeline while approximately preserving each period’s default rate.

### Why split by time?

Our intended use is to learn from older loans and score newer applications. A time split follows this direction and allows us to evaluate performance on a later cohort.

### Why exclude 2017–2018?

Only resolved loans are included in the source dataset. Recent cohorts may contain a selective subset of loans that reached a final outcome before data collection.

Excluding these years reduces this concern, but does not eliminate outcome-selection bias.

### Test-set disclosure

The split notebook already displayed the test set’s overall default rate. We therefore describe it as a reserved test set whose aggregate label distribution has been inspected.

Individual test observations and model results must not be used for feature selection, tuning, or threshold selection.

### Timing limitation

We do not know when each final loan outcome became available. Some loans issued before 2016 may have resolved after 2016.

Our split is therefore a retrospective comparison across origination periods, rather than a complete simulation of information available at a historical deployment date.

## 5. Exploratory Data Analysis

**Owner: Data Exploration teammate**

The EDA notebook profiles each variable’s:

- Type
- Missing values
- Noise and outliers
- Distribution
- Potential usefulness

All EDA uses the training sample.

### Variable summary

| Variable | Meaning | Main finding / planned treatment |
|---|---|---|
| `id` | Loan identifier | Check overlap; exclude from predictors |
| `issue_d` | Issue month and year | Use for splitting and yearly analysis |
| `revenue` | Self-reported annual income | Strongly skewed; evaluate clipping and log transformation |
| `dti_n` | Monthly debt-to-income ratio, excluding mortgage, as defined by the source | Retain; evaluate extreme values |
| `loan_amnt` | Requested loan amount | Retain; use in loan-to-income |
| `fico_n` | Midpoint of reported FICO range | Retain; use in the baseline |
| `experience_c` | Borrower-experience indicator | Nearly constant; exclude |
| `emp_length` | Employment length | Preserve information from `NI` |
| `purpose` | Loan purpose | Encode; review rare categories |
| `home_ownership_n` | Housing status | Encode; review rare categories |
| `addr_state` | State of residence | Review sparse categories and geographic dependence |
| `zip_code` | Partially masked ZIP code | Exclude from baseline |
| `title` | Borrower-provided loan title | Exclude from structured baseline |
| `desc` | Borrower-provided loan description | Exclude because of missingness and changing availability |
| `Default` | Charged off or fully paid | Target |

### Main findings

- About **18.5%** of training loans defaulted.
- Income has skew around **46.4** and a maximum of **USD 9.5 million**.
- A log transformation substantially reduces income skew.
- About **5.2%** of borrowers have `emp_length = NI`.
- The `NI` group has an observed default rate of approximately **23.7%**.
- Only **5 of 150,000** training observations have `experience_c = 0`.
- ZIP code has **879 categories**; approximately **20%** have fewer than 30 observations.
- Loan descriptions are approximately **86% missing** and are almost absent in 2015.
- Only **90 training loans** have FICO below 660, so evidence for this population is limited.

These statistics come from the supplied notebook outputs.

### Charts and takeaways

| Chart | Takeaway from training EDA |
|---|---|
| Default rate by FICO | Higher FICO bands generally have lower default rates; the lowest band has very few observations |
| Default rate by purpose | Small-business loans have higher observed default rates than car loans |
| Default rate by home ownership | Renters have higher observed default rates than mortgage holders |
| Default rate by year | Rates vary over time and rise in the later training years |
| Default rate by loan-to-income | Higher ratio groups have higher observed default rates |

Each chart should include readable labels, relevant group counts, and a one-line takeaway.

These relationships are descriptive. They do not establish causation, and weak correlations do not prove that the data are free of leakage.

### Additional data that would help

| Additional data | Why it would help |
|---|---|
| Income-verification information | Assess reporting reliability |
| Dates when outcomes became known | Create a more realistic historical evaluation |
| Recent cohorts with sufficient follow-up | Assess current relevance |
| Loan returns, funding costs, and recoveries | Evaluate financial value |
| Broader application records | Understand coverage beyond approved applicants |
| Appropriate protected-group information | Support a reviewed subgroup fairness assessment |

Application records alone would not reveal how rejected applicants would have repaid. That question would still require careful treatment.

## 6. Data Preparation and Shared Pipeline

**Owner: Data Preparation & Pipeline teammate / Repo Owner**

Everyone will reuse a shared sklearn preprocessing pipeline. Model-specific versions may differ where scaling or encoding requirements differ, but comparisons should use consistent data and validation rules.

The cross-reviewer will check that validation and test information do not enter training.

### Planned steps

1. Drop identifiers and excluded features.
2. Handle missing and unusual values.
3. Engineer candidate features.
4. Encode categorical variables.
5. Scale numeric variables for models that need it.
6. Fit the model.

### Candidate features and transformations

| Feature / transformation | Reason |
|---|---|
| `loan_to_income = loan_amnt / revenue` | Capture requested loan size relative to annual income |
| Log income | Reduce strong right skew |
| FICO bands | Compare a grouped representation with continuous FICO |
| Employment missingness category or flag | Preserve information from `NI` |
| Income and DTI clipping | Evaluate sensitivity to extreme observations |
| Rare-category grouping | Avoid unstable estimates from very small categories |

Loan-to-income is not a monthly repayment burden or a complete affordability measure. Invalid or nonpositive income must be handled explicitly.

### DTI assumption

The supplied training EDA reports a maximum DTI of **83.4**. It does not verify that 999 appears in the raw file or is an official missing-value code.

If 999 appears during later processing, investigate its meaning and document any assumption. Values above 100 should not automatically be treated as impossible.

### Leakage prevention

The following must be fitted using each training fold only:

- Imputation values
- Clipping limits
- Scaling parameters
- Category vocabularies
- Rare-category grouping rules
- Any feature-selection procedure

If target encoding is used, it must also avoid using a row’s own target to encode that row.

**Final implemented pipeline:** TBD.

## 7. Evaluation Metrics

### ROC-AUC

ROC-AUC measures how well the model ranks defaulted loans above fully paid loans across thresholds.

We use it as the primary ranking metric. It does not determine an approval threshold or show whether predicted probabilities are calibrated.

### Precision–recall performance

We also report **Average Precision (AP)** using:

`sklearn.metrics.average_precision_score`

AP summarizes precision–recall performance for the default class, which is the minority class. We will also show a precision–recall curve.

For consistency, we label the number **AP**. Its calculation differs from trapezoidal PR-AUC.

The no-skill precision reference is the default prevalence of the evaluated set, so prevalence should be considered when comparing periods.

### Approval rate versus default rate

For a default-risk score, approve a loan when its score is at or below a selected threshold.

`Approval rate = approved loans / evaluated loans`

`Approved-loan default rate = defaults among approved loans / approved loans`

If no loans are approved, the approved-loan default rate is undefined.

These measures connect model performance to our business question. At a similar approval rate, a useful model should retain a lower-risk group than the baseline.

Because the dataset contains historical issued loans, “approval rate” means the fraction retained from that population. It does not estimate approval rates for all applicants.

### Why not accuracy alone?

Predicting every training loan as fully paid would achieve approximately **81.5% accuracy** while identifying no defaults.

### Supporting diagnostics

We will also examine:

- Confusion matrices at validation-selected thresholds
- Default precision and recall
- Approved counts
- Calibration if scores are interpreted as probabilities

## 8. Baseline, Model Comparison and Shortlist

**Owner: Baseline & Shortlist teammate**  
**Status: Results pending**

### Models

Build a FICO-only logistic regression baseline, then compare 4–5 model families, such as:

- Logistic regression
- Decision tree
- Random forest
- Gradient boosting
- kNN

The final set should reflect available time and computing resources.

### Cross-validation plan

Our team plan uses **stratified 5-fold CV within the training set**.

- Use the same folds across models.
- Use shuffling with random seed **42**.
- Fit learned preprocessing separately within each fold.
- Report ROC-AUC and AP as **mean ± standard deviation**.

Stratified CV keeps class proportions similar across folds, but it mixes origination years. Its results do not directly measure future-cohort performance.

The separate 2016 test set provides the later-period comparison. A chronological validation check would provide additional evidence if time allows.

### Model comparison results

| Model | CV ROC-AUC, mean ± SD | CV AP, mean ± SD | Runtime | Main observation |
|---|---|---|---|---|
| FICO-only logistic regression | TBD | TBD | TBD | Baseline |
| Multivariable logistic regression | TBD | TBD | TBD | TBD |
| Decision tree | TBD | TBD | TBD | TBD |
| Random forest | TBD | TBD | TBD | TBD |
| Gradient boosting | TBD | TBD | TBD | TBD |
| kNN, if included | TBD | TBD | TBD | TBD |

The course rubric asks for a shortlist of **3–5 promising candidates**. We will document this shortlist, then prioritize the strongest **2–3** for detailed tuning, following our team plan.

Selection should consider performance, stability, runtime, interpretability, and error patterns.

**Shortlist and reasons:** TBD.

## 9. Feature Importance, Error Analysis and Iteration

**Owner: Baseline & Shortlist teammate**  
**Status: Pending model predictions**

### Feature importance

Use suitable methods to explain which variables each model relies on.

Feature importance describes model behaviour. It does not establish that changing a feature would change a borrower’s outcome.

### Error analysis

With default defined as the positive class:

- **False approval:** approve a loan that defaults — a false negative.
- **False rejection:** reject a loan that repays — a false positive.

We will examine errors by:

- FICO range
- DTI
- Loan-to-income
- Loan purpose
- Employment-information availability
- Origination year

Report group sizes and avoid strong conclusions from very small groups.

We will also discuss what additional information a human reviewer might have needed, such as verified income or more detailed repayment obligations.

### Feature and preprocessing experiments

| Experiment | Why we tried it | Validation result | Decision |
|---|---|---|---|
| Income transformation | Reduce sensitivity to skew | TBD | TBD |
| Loan-to-income | Capture relative requested exposure | TBD | TBD |
| FICO bands | Test a different representation | TBD | TBD |
| Employment missingness treatment | Preserve missingness information | TBD | TBD |
| State inclusion versus exclusion | Assess predictive value and geographic dependence | TBD | TBD |

## 10. Fine-tuning and Ensemble

**Owner: Fine-tuning & Final Evaluation teammate**  
**Status: Pending**

### Hyperparameter tuning

Use `RandomizedSearchCV` on the training data for selected candidates.

Record:

- Search ranges
- Number of sampled configurations
- CV settings
- Random seed
- Primary scoring metric
- Best settings
- Validation results

Use ROC-AUC as the primary search metric and report AP as well. Randomized search allows us to explore broader parameter spaces within the project timeline.

Uncertain preprocessing choices should also be compared using validation data.

### Ensemble

Try an ensemble of suitable models and compare it with the best individual model.

An ensemble does not have to win. We will report whether its performance justifies the extra complexity.

| Candidate | Search method / budget | Best settings | Validation result | Decision |
|---|---|---|---|---|
| Candidate 1 | TBD | TBD | TBD | TBD |
| Candidate 2 | TBD | TBD | TBD | TBD |
| Candidate 3 | TBD | TBD | TBD | TBD |
| Ensemble | TBD | TBD | TBD | TBD |

## 11. Threshold Selection and Final Evaluation

**Owner: Fine-tuning & Final Evaluation teammate**  
**Status: Pending**

### Policy selection

Use a held-out portion of training data or clearly documented out-of-fold predictions to select an approval policy.

Do not select thresholds using in-sample fitted predictions or final test outcomes.

If the policy approves the safest X%, define the approval fraction and tie-handling rule before examining test results.

### Final evaluation procedure

1. Select the model and hyperparameters using development validation.
2. Select the approval policy using validation predictions.
3. Freeze preprocessing, model settings, and policy.
4. Refit the selected procedure on the training data.
5. Evaluate the final model and baseline on the reserved test set.
6. Do not tune further in response to test results.

### Predictive results

| Model | Test ROC-AUC | Test AP | Main conclusion |
|---|---|---|---|
| FICO-only baseline | TBD | TBD | TBD |
| Final model | TBD | TBD | TBD |

### Business results

| Policy | Test approval rate | Default rate among approved loans | Approved count |
|---|---|---|---|
| Retain all historical test loans | 100% | 23.28% | 50,000 |
| Baseline policy selected on validation | TBD | TBD | TBD |
| Final policy selected on validation | TBD | TBD | TBD |

The retain-all row is a descriptive reference, not a model result.

Our final business statement will use actual results:

> At an approval rate of **TBD%**, the final model retained loans with an observed default rate of **TBD%**, compared with **TBD%** for the baseline at a similar approval rate.

We will use the actual test reference rate of **23.28%**, rather than an illustrative 20%.

## 12. Why We Exclude ZIP Code

We exclude ZIP code from the baseline for technical and ethical reasons.

### Technical reason

Many ZIP-code categories contain few observations. This can produce unstable estimates and increase overfitting risk.

### Ethical reason

Residential location can act as a proxy for race or ethnicity. Historical redlining shows how lending restrictions based on neighbourhood characteristics can harm communities.

Dropping ZIP code reduces reliance on detailed residential geography.

### Remaining limitations

Removing ZIP code does not guarantee fairness or legal compliance. State, housing, income, and credit history may also contain proxy information.

We do not have direct race or ethnicity labels, so we cannot claim to have measured racial fairness.

Real use would require further review. We also do not claim that every use of ZIP code is automatically unlawful.

## 13. What Worked and What Did Not

### What we know so far

- Stratified sampling approximately preserved each selected period’s default rate.
- EDA identified important issues, including `NI`, skewed income, and sparse categories.
- Log transformation reduced income skew, but its effect on model performance remains untested.

### Design choices

We ruled out accuracy as the only metric and excluded ZIP code, descriptions, titles, and the nearly constant experience feature from the structured baseline.

These are design choices, not evidence that alternative trained models performed worse.

### Modelling lessons

Each modelling teammate will add lessons from actual experiments, including attempts that did not improve validation performance.

| Attempt | Expected benefit | Observed result | What we learned |
|---|---|---|---|
| TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD |
| Ensemble | Combine model strengths | TBD | TBD |

We will not describe an approach as unsuccessful without supporting results.

## 14. Assumptions and Limitations

| Issue | Why it matters | Response / next step |
|---|---|---|
| Only approved loans are observed | Results may not apply to rejected applicants | Limit conclusions to the observed population |
| Only resolved loans are retained | Cohorts may be selective | State the bias and seek better follow-up information |
| Outcome dates are unavailable | Time splitting may not match information available at deployment | Obtain outcome dates for a more realistic evaluation |
| Stratified CV mixes years | Validation results may hide temporal changes | Retain the later-year test and consider a chronological check |
| Income is self-reported | Inputs may contain reporting errors | Evaluate transformations and seek verification information |
| Few loans have FICO below 660 | Performance is uncertain for this population | Avoid unsupported extrapolation |
| Data are historical | Current lending conditions may differ | Validate on more recent cohorts with sufficient follow-up |
| ZIP removal does not ensure fairness | Other features may still act as proxies | Review remaining features and subgroup outcomes |
| Probability calibration is unverified | Scores may not represent actual default probabilities | Assess calibration before interpreting probabilities |
| Costs and recoveries are missing | Lower default frequency does not prove higher profit | Avoid profit claims and seek financial data |
| Overall test prevalence was inspected | The holdout was not completely untouched | Disclose this and keep remaining decisions separate |

## 15. How to Run the Project

### Current notebooks

| File | Purpose | Status |
|---|---|---|
| `01_split_and_sample(1).ipynb` | Time split and stratified sampling | Completed |
| `02_eda.ipynb` | Training-data exploration | Completed |
| Shared pipeline notebook/module | Reusable preprocessing | TBD |
| Model comparison notebook | Baseline, CV and error analysis | TBD |
| Tuning and final evaluation notebook | Search, ensemble and test results | TBD |

### Execution order

1. Download `LC_loans_granting_model_dataset.csv` from Zenodo.
2. Place the CSV where notebook 01 expects it.
3. Run notebook 01 to create `train.csv` and `test.csv`.
4. Run notebook 02 on `train.csv`.
5. Run the pipeline and modelling notebooks once available.
6. Run final evaluation after model and policy choices are frozen.

The current split and EDA notebooks use the working directory for the sample CSV files. If the repo owner moves data into a `data/` folder, update paths consistently.

### Environment

- **Python version:** TBD
- **Package versions / requirements:** TBD
- **Sampling seed:** 42
- **CV seed:** 42
- **Other experiment seeds:** document in the relevant notebooks

The repo owner will finalize folder paths, dependencies, and execution instructions.

Check data redistribution terms before committing raw or derived data.

## 16. Presentation Plan — 10 Minutes

| Time | Speaker | Main content |
|---|---|---|
| 0:00–2:00 | Project Lead | Business problem, users, dataset fit, metrics and scope |
| 2:00–4:00 | EDA | Main data quirks and selected charts with takeaways |
| 4:00–6:00 | Pipeline | Time split, shared preprocessing, engineered features and leakage checks |
| 6:00–8:00 | Baseline & Shortlist | CV results, important variables, errors and lessons |
| 8:00–10:00 | Fine-tuning & Evaluation | Final results, approval/default trade-off, limitations and conclusion |

The project lead collects GitHub IDs, combines slides, checks transitions, and organizes rehearsal.

Each person prepares and presents their own section. We will agree on the final conclusion together.

### Presentation rules

- Start with the business problem.
- Use labelled, readable charts.
- Give each chart one clear takeaway.
- Avoid pasting large notebook outputs onto slides.
- Explain both successful and unsuccessful experiments.
- End with what the results support and where the evidence stops.
- Keep the total presentation within 10 minutes.

## 17. Eight-Day Plan

| Days | Focus | Expected output |
|---|---|---|
| 1–2 | Framing, repo setup, EDA, split, pipeline skeleton and baseline | Shared data, initial README, EDA findings and runnable baseline |
| 3–5 | Feature engineering and model comparison | CV tables, experiment log, error analysis and shortlist |
| 6–7 | Tuning, ensemble, final evaluation, README and slides | Frozen final model, results, completed documentation and slide draft |
| 8 | Cross-review, rehearsal and final repo check | Reproducible repository and presentation within 10 minutes |

## 18. Final Submission Checklist

- [ ] Everyone’s name and GitHub ID are recorded.
- [ ] Each variable is profiled in the EDA notebook.
- [ ] Each chart has labels and a takeaway.
- [ ] The shared pipeline has been cross-reviewed for leakage.
- [ ] Model comparisons include CV mean and standard deviation.
- [ ] The shortlist and selection reasons are documented.
- [ ] Feature importance and error analysis are included.
- [ ] Hyperparameter tuning and ensemble results are recorded.
- [ ] Threshold selection is separate from final test evaluation.
- [ ] Actual successful and unsuccessful attempts are documented.
- [ ] Assumptions include likely impacts and next steps.
- [ ] Results and instructions match the final code.
- [ ] Dependencies and execution order are recorded.
- [ ] Result placeholders are replaced before submission.
- [ ] Slides fit the 10-minute limit.

## 19. Current Conclusion

The completed data split and EDA provide a starting point for comparing default-risk models.

The next step is to test whether additional application features improve risk ranking and the approval–default trade-off relative to the FICO-only baseline.

Our final recommendation will depend on validation-selected procedures and reserved-test results. We will interpret those results within the limits of historical approved-loan data.

## References

- [Lending Club loan dataset for granting models, version 0.1](https://doi.org/10.5281/zenodo.11295916)