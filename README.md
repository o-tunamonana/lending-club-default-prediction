# Loan Default Prediction: Balancing Approval and Risk

**Course:** INSY 674  
**Status:** Data preparation, EDA, model comparison, tuning, and final evaluation completed.  
**Dataset:** [Lending Club loan dataset for granting models](https://zenodo.org/records/11295916)

## 1. Project Overview

Lending decisions involve two competing goals: approving enough loans to maintain lending volume while limiting defaults. A stricter policy can reduce risk, but it also rejects borrowers who would have repaid.

Our main question is:

> If we approve the lowest-risk loans, how does the default rate change as we approve more or fewer loans?

We compare five model families against a FICO-only logistic regression baseline using application-time information. We evaluate model scores alongside approval rates and default rates among approved loans.

### Main result

On the 2016 test sample, approving the lowest-risk **80%** using tuned gradient boosting produced:

- **19.50%** default rate among approved loans.
- **21.56%** with FICO alone at the same approval rate.
- **23.28%** across all test loans.

At the same volume of **40,000 approved loans**, gradient boosting retained **827 fewer loans that ultimately defaulted** than the baseline.

Risk ranking remained similar across CV and the later test cohort. However, the model underestimated 2016 default probabilities, making calibration an important remaining issue.

## 2. Business Objective and Problem Framing

### Business objective

Reduce defaults among approved loans while retaining a meaningful share of lending volume.

A policy that approves very few loans may achieve a low default rate without meeting lending-volume goals. We therefore compare models at matched approval rates and examine how risk changes as approval increases.

We ask:

- Does the model rank loan risk better than FICO alone?
- At the same approval rate, does it retain a lower-risk group?
- How many defaults are rejected, and how many fully paid loans are also rejected?

Financial returns, funding costs, and recoveries are unavailable, so we measure the approval–default trade-off rather than profit.

### Intended users and use

Credit-risk analysts could use scores to compare risk and investigate errors. Lending-policy teams could use approval curves to compare alternative policies.

The workflow is to collect application information, apply the fitted pipeline, generate risk scores, and apply a policy selected using development data.

### Machine learning task

This is a **supervised binary classification** problem.

- **Unit of analysis:** one loan.
- **Target:** `Default`.
- **Positive class, 1:** charged off.
- **Negative class, 0:** fully paid.
- **Training:** offline, using historical loans.
- **Output:** a score where higher values indicate higher default risk.

### Baseline

Manual underwriting and credit-score rules are common lending approaches. We use **FICO-only logistic regression** as a simple benchmark for assessing the value of additional application features.

## 3. Team Members

| Member | Student ID | Role | Main responsibilities |
|---|---|---|---|
| Dora Zhuang | 261012351 | Project Lead & Storyteller | Business framing, metrics, README, assumptions, limitations, slides and presentation flow |
| Sirui Chen | 261342345 | Data Exploration | Variable profiles, data quirks, charts and takeaways |
| Una Qian | 260969447 | Data Preparation & Pipeline / Repo Owner | Time split, sampling, shared pipeline, repo structure and merges |
| Bingqiang Zhao | 261350698 | Baseline & Shortlist | Baseline, model comparison, feature importance and error analysis |
| Nat Suriyapanuwat | 261345843 | Fine-tuning & Final Evaluation | Hyperparameter search, ensemble, threshold selection and final test evaluation |

## 4. Data Source and Split

The [source dataset](https://doi.org/10.5281/zenodo.11295916) contains **1,347,681 loans and 15 columns**, covering origination years **2007–2018**. It retains fully paid and charged-off loans and excludes transitional outcomes.

Rejected applications are absent. Our analysis therefore concerns historical issued loans with resolved outcomes.

### Train/test split

We separate origination periods before EDA and fitted preprocessing, then take stratified samples.

| Dataset | Issue years | Available pool | Sample size | Default rate |
|---|---|---:|---:|---:|
| Training / development | 2007–2015 | 829,347 | 150,000 | 18.46% |
| Test | 2016 | 293,057 | 50,000 | 23.28% |

Sampling is stratified by `Default`, with seed **42**. Sampled train and test IDs do not overlap. EDA uses the training sample.

Sampling makes repeated comparisons manageable while approximately preserving each period’s default prevalence.

### Why split by time?

The intended task is to learn from older loans and score newer applications. The 2016 test cohort provides a later-period comparison, with observed default prevalence increasing from **18.46%** to **23.28%**.

### Why exclude 2017–2018?

Only resolved loans remain in the dataset, so recent cohorts may overrepresent loans that reached a final outcome quickly. We exclude these years to reduce this outcome-selection concern.

### Evaluation boundaries

Aggregate test prevalence was inspected in the split notebook, and the pipeline notebook documents test-period DTI anomalies. The test sample was therefore not completely unseen during development.

Model fitting, hyperparameter search, and approval-fraction selection use training data. Notebook 05 reports that its model and policy choices were fixed before final test evaluation.

Outcome-availability dates are missing. Some earlier loans may have resolved after 2016, so the split compares origination cohorts rather than fully reconstructing historically available training labels.

## 5. Exploratory Data Analysis

[Notebook 02](notebooks/02_eda.ipynb) profiles each variable’s type, missingness, unusual values, distribution, and usefulness.

### Variable summary

| Variable | Meaning | Treatment |
|---|---|---|
| `id` | Loan identifier | Overlap checks; excluded from predictors |
| `issue_d` | Issue month and year | Temporal splitting and cohort analysis |
| `revenue` | Self-reported annual income | Log transformation and standardization |
| `dti_n` | Monthly debt-to-income ratio, excluding mortgage, as defined by the source | Cleaning, median imputation and standardization |
| `loan_amnt` | Requested loan amount | Log transformation and standardization |
| `fico_n` | Midpoint of reported FICO range | Median imputation and standardization |
| `experience_c` | Borrower-experience indicator | Excluded because it is nearly constant |
| `emp_length` | Employment length | Numeric years, imputation and missingness indicator |
| `purpose` | Loan purpose | One-hot encoding with rare-category grouping |
| `home_ownership_n` | Housing status | One-hot encoding with rare-category grouping |
| `addr_state` | State of residence | Smoothed target encoding with cross-fitting |
| `zip_code` | Partially masked ZIP code | Excluded |
| `title` | Borrower-provided loan title | Excluded because of high cardinality and overlap with purpose |
| `desc` | Borrower-provided loan description | Excluded because of missingness and changing availability |
| `Default` | Charged off or fully paid | Target |

### Main findings

- Approximately **18.5%** of training loans defaulted.
- Income has skew around **46.4** and a maximum of **USD 9.5 million**.
- Log transformation reduces income skew to approximately **0.18**.
- About **5.2%** have `emp_length = NI`, with an observed default rate of approximately **23.7%**.
- Only **5 of 150,000** training observations have `experience_c = 0`.
- ZIP code has **879 categories**; approximately **20%** have fewer than 30 observations.
- Loan descriptions are approximately **86% missing** and almost absent in 2015.
- Only **90 training observations** have FICO below 660.

### Charts and takeaways

| Chart | Main takeaway |
|---|---|
| Default rate by FICO | Higher FICO bands generally have lower default rates |
| Default rate by purpose | Small-business loans have higher observed default rates than car loans |
| Default rate by home ownership | Renters have higher observed default rates than mortgage holders |
| Default rate by year | Rates vary over time and rise in later training years |
| Default rate by loan-to-income | Higher ratio groups have higher observed default rates |

Individual numeric rank correlations with default are weak. We therefore test combinations of features and evaluate their contribution through validation.

### Useful additional data

| Additional data | Purpose |
|---|---|
| Income verification | Assess reporting reliability |
| Dates when outcomes became known | Reconstruct historically available labels |
| Recent cohorts with sufficient follow-up | Assess current relevance |
| Loan returns, costs and recoveries | Evaluate financial value |
| Broader application records | Understand coverage beyond approved applicants |
| Appropriate protected-group information | Assess approval and error disparities directly |

## 6. Preprocessing and Feature Engineering

[Notebook 03](notebooks/03_pipeline.ipynb) explains the shared pipeline in [preprocessing.py](notebooks/preprocessing.py).

Full models use eight original application inputs and the engineered `loan_to_income` feature.

### Cleaning

- Convert employment length to numeric years from 0 to 10.
- Map `NI` to missing and preserve missingness through an indicator.
- Treat DTI equal to 999 as missing.
- Cap DTI at 100.
- Calculate `loan_to_income = loan_amnt / revenue`.
- Drop identifiers, issue date, experience indicator, ZIP code, title and description.

The source does not explain DTI equal to 999. Treating it as missing is an explicit assumption. The training sample has no such values and no DTI above 100; these rules handle later-period observations.

### Transformations

| Group | Features | Transformation |
|---|---|---|
| Numeric | FICO, DTI, employment length | Median imputation with missing indicators, then standardization |
| Skewed numeric | Income, loan amount, loan-to-income | `log(1 + x)`, then standardization |
| Categorical | Purpose, housing status | One-hot encoding; categories with fewer than 500 observations grouped as infrequent |
| Geographic | State | Target encoding with smoothing of 100 and internal cross-fitting |

Income percentile clipping and FICO bands were not included in the final shared pipeline.

The full-training pipeline check produces **24 numeric columns**, with no remaining missing values.

### Leakage control

Learned preprocessing is fitted inside each CV training fold. State target encoding uses internal cross-fitting so a row’s own target does not determine its training encoding.

The FICO-only baseline uses a separate pipeline with FICO imputation and scaling.

## 7. Evaluation Metrics

### ROC-AUC

ROC-AUC measures how well a model ranks defaulted loans above fully paid loans. It is the primary metric for model comparison and hyperparameter selection.

### Average Precision

We report **Average Precision (AP)** using `average_precision_score`, alongside precision–recall analysis.

AP focuses on default identification in an imbalanced dataset. The prevalence reference is **18.46%** in training and **23.28%** in testing. AP differs from trapezoidal PR-AUC, so we label it explicitly.

### Approval and default rates

`Approval rate = approved loans / evaluated loans`

`Approved-loan default rate = defaults among approved loans / approved loans`

Models are compared at the same approval rate, so lower risk is not achieved simply by approving fewer loans. Approval here means retaining loans from the historical issued-loan sample.

### Supporting diagnostics

We examine confusion matrices, default precision and recall, Brier score, and calibration.

Accuracy alone is insufficient: predicting every training loan as fully paid achieves approximately **81.5% accuracy** while identifying no defaults.

## 8. Model Comparison and Shortlist

[Notebook 04](notebooks/04_modeling_baseline.ipynb) compares five model families against the FICO-only baseline.

All models use **stratified five-fold CV**, with shuffling and seed **42**. Learned preprocessing is fitted separately within each fold. Results are mean ± sample standard deviation.

| Model | CV ROC-AUC | CV AP |
|---|---|---|
| Gradient boosting | 0.6627 ± 0.0047 | 0.3007 ± 0.0084 |
| Logistic regression | 0.6610 ± 0.0047 | 0.2965 ± 0.0083 |
| Random forest | 0.6608 ± 0.0041 | 0.2970 ± 0.0065 |
| Extra Trees | 0.6554 ± 0.0039 | 0.2930 ± 0.0087 |
| Decision tree | 0.6420 ± 0.0044 | 0.2736 ± 0.0057 |
| FICO-only baseline | 0.5904 ± 0.0043 | 0.2254 ± 0.0028 |

All multivariable models outperform the FICO-only baseline.

We shortlisted:

1. **Gradient boosting:** highest mean ROC-AUC and AP.
2. **Logistic regression:** similar performance with a simpler model.
3. **Random forest:** a competitive alternative ensemble approach.

The small differences between these candidates motivate examining tuning gains and complexity.

Shuffled CV mixes origination years. The separate 2016 test provides the later-cohort evaluation.

## 9. Feature Importance and Error Analysis

### Important variables

Held-out permutation importance uses up to 3,000 observations from the first validation fold and five permutations per feature.

For gradient boosting:

| Original input | Mean ROC-AUC reduction after permutation |
|---|---:|
| FICO | 0.0673 |
| Loan amount | 0.0657 |
| Income | 0.0340 |
| DTI | 0.0264 |

Shuffling loan amount or income also changes the derived loan-to-income feature. These values measure predictive contribution through the pipeline, rather than causal effects.

### Default threshold

At the preliminary threshold of **0.50**, untuned gradient boosting identified only **82 of 27,684** training defaults in out-of-fold predictions.

Most scores fall below 0.50. This produces high accuracy but very low default recall, motivating a policy based on approval volume.

### Business errors

With default as the positive class:

- **False approval:** approve a loan that defaults — false negative.
- **False rejection:** reject a loan that repays — false positive.

Notebook 04 examines errors across FICO, predicted risk, housing, purpose, employment information, and other loan characteristics. Verified income and more detailed repayment obligations would help investigate individual errors.

## 10. Hyperparameter Tuning and Ensembles

[Notebook 05](notebooks/05_tuning_and_final_evaluation.ipynb) tunes the three shortlisted models using `RandomizedSearchCV` and the same five-fold design.

### Search budgets

| Model | Configurations | Main parameters |
|---|---:|---|
| Gradient boosting | 40 | Learning rate, iterations, leaf count, minimum leaf size, regularization, feature fraction |
| Random forest | 20 | Tree count, depth, minimum leaf size, feature fraction |
| Logistic regression | 15 | Regularization strength and L1/L2 penalty |

ROC-AUC determines the selected configuration; AP is also recorded. Randomized search explores broad ranges within a manageable number of fits.

### Tuning results

| Model | Untuned CV ROC-AUC | Tuned CV ROC-AUC | Tuned AP |
|---|---:|---|---:|
| Gradient boosting | 0.6627 | 0.6636 ± 0.0046 | 0.3012 |
| Random forest | 0.6608 | 0.6612 ± 0.0039 | 0.2973 |
| Logistic regression | 0.6610 | 0.6610 ± 0.0047 | 0.2966 |

The largest tuning gain is approximately **0.0009 ROC-AUC**. Search adds little compared with the improvement from using more information than FICO alone.

### Final gradient boosting configuration

The final estimator is `HistGradientBoostingClassifier`, with settings rounded here for readability:

| Parameter | Value |
|---|---:|
| `learning_rate` | 0.0249 |
| `max_iter` | 510 |
| `max_leaf_nodes` | 15 |
| `min_samples_leaf` | 337 |
| `l2_regularization` | 0.0257 |
| `max_features` | 0.5823 |

Exact sampled settings are saved by notebook 05. Early stopping can stop fitting before the maximum iteration count.

### Ensembles

We tested soft voting and stacking with a logistic-regression meta-model and internal five-fold CV.

Both achieved approximately **0.6642 CV ROC-AUC**, versus **0.6636** for tuned gradient boosting, without an AP improvement. Component predictions were highly correlated, approximately **0.93–0.97**.

We retained gradient boosting because the small gain did not justify running three models. The notebook required a gain larger than one fold standard deviation as a practical complexity rule, not a significance test.

## 11. Approval Policy and Final Test Results

### Policy selection

Using out-of-fold training predictions, we selected **80% approval** before final test evaluation.

We evaluated:

1. **Fixed approval fraction:** approve the lowest-risk 80% of the scored batch.
2. **Fixed training cutoff:** apply the threshold derived from training predictions.

Score ties in the fixed-fraction policy are broken randomly with seed **42**, particularly important for the FICO baseline.

At 80% training approval, approved-loan default rates were **15.06%** for gradient boosting and **17.08%** for FICO. Training-derived cutoffs were **0.2565** and **0.2281**, respectively.

The final model and baseline were fitted on all 150,000 training loans. Notebook 05 reports that predictions were saved after final test scoring and no further tuning followed.

### Predictive performance

| Model | Test ROC-AUC | Test AP | Test Brier |
|---|---:|---:|---:|
| FICO-only baseline | 0.5943 | 0.2822 | 0.1774 |
| Tuned gradient boosting | 0.6639 | 0.3598 | 0.1706 |

Gradient boosting improves test ROC-AUC by approximately **0.0696**. Its test score is close to its **0.6636** tuned CV result, showing similar ranking performance on the later cohort.

### Approval and default rates

| Policy | Approval rate | Approved loans | Approved-loan default rate |
|---|---:|---:|---:|
| Retain all test loans | 100.00% | 50,000 | 23.28% |
| FICO-only, safest 80% | 80.00% | 40,000 | 21.56% |
| Gradient boosting, safest 80% | 80.00% | 40,000 | 19.50% |
| FICO-only, training cutoff | 81.24% | 40,621 | 21.70% |
| Gradient boosting, training cutoff | 80.07% | 40,036 | 19.50% |

At matched 80% approval, gradient boosting lowers the default rate by approximately **2.07 percentage points** relative to FICO alone.

At 50% approval, test default rates are **14.86%** for gradient boosting versus **18.40%** for FICO. At 90%, they are **21.08%** versus **22.48%**. Gradient boosting has lower observed default rates at each evaluated approval fraction from 10% to 90%.

### Errors at 80% approval

| Outcome | Gradient boosting | FICO-only |
|---|---:|---:|
| Approved and fully paid | 32,201 | 31,374 |
| Approved and defaulted | 7,799 | 8,626 |
| Rejected and fully paid | 6,158 | 6,985 |
| Rejected and defaulted | 3,842 | 3,015 |

Gradient boosting rejects **33.0%** of defaults, compared with **25.9%** for FICO.

However, **61.58%** of rejected loans were fully paid. Better ranking reduces errors relative to the baseline but still excludes many borrowers who repay.

### Bootstrap uncertainty

Notebook 05 uses **1,000 paired bootstrap resamples** of the test sample. The 95% percentile interval for the approved-loan default-rate difference, gradient boosting minus FICO at 80% approval, is approximately **−2.37 to −1.85 percentage points**.

This describes uncertainty within the observed test sample, not retraining variability or future changes in lending conditions.

## 12. Calibration and Temporal Change

Ranking remained stable, but calibration changed.

On training out-of-fold predictions, predicted and observed default rates were within approximately **0.7 percentage points** across ten risk groups.

On the 2016 test sample:

- Mean predicted default probability: **18.49%**
- Actual default prevalence: **23.28%**

The model underestimated defaults across all ten test risk groups.

Risk ranking and probability accuracy therefore need separate evaluation. The model identifies relatively safer loans, while its numerical probabilities understate the later cohort’s default level.

A next step is to assess recalibration using a recent labelled validation cohort, followed by evaluation on another untouched period.

## 13. Fairness and ZIP-Code Exclusion

ZIP code is excluded for statistical and ethical reasons.

### Sparse categories

Training contains **879 ZIP-code categories**, with approximately **20%** represented by fewer than 30 loans. Small groups can produce unstable estimates.

### Geographic proxies

Residential geography can act as a proxy for race or ethnicity. Historical redlining illustrates how neighbourhood-based lending restrictions can exclude communities from credit.

We remove ZIP code to reduce dependence on detailed residential location when assessing individual borrowers.

State remains through smoothed target encoding. A useful extension is to compare model performance and approval patterns with and without state.

Race and ethnicity labels are unavailable, so racial approval and error disparities cannot be measured directly. ZIP-code exclusion addresses one feature choice; it does not establish fair model outcomes.

## 14. What Worked and What Did Not

### What worked

- **Additional application features:** multivariable models improved substantially on FICO alone.
- **Approval-based evaluation:** matched approval rates connected ranking performance to a lending decision.
- **Later-cohort ranking:** test ROC-AUC remained close to CV performance.
- **Missingness handling:** the pipeline retained employment-information missingness identified in EDA.

### What added little or did not transfer

| Attempt / finding | Result | Lesson |
|---|---|---|
| Default threshold of 0.50 | Very low default recall | Thresholds should reflect the lending objective |
| Hyperparameter search | Largest ROC-AUC gain below 0.001 | Broader tuning produced limited gains |
| Soft voting and stacking | Approximately +0.0006 ROC-AUC; no AP gain | Correlated models added little diversity |
| Loan-to-income | Notebook 03 reports similar boosting performance with and without it | A strong EDA relationship may duplicate existing information |
| Training-period calibration | Underestimated 2016 defaults | Stable ranking does not imply stable probabilities |

The description-availability flag was removed because descriptions became almost absent in later years. Income percentile clipping and FICO bands were not included in the final pipeline.

## 15. Assumptions and Limitations

| Issue | Effect on interpretation | Next step |
|---|---|---|
| Only issued loans are observed | Results may differ for rejected applicants | Obtain broader application data and assess coverage |
| Only resolved outcomes are retained | Cohorts may be selective | Seek fuller follow-up information |
| Outcome dates are unavailable | Time splitting cannot fully reproduce historical label availability | Construct an as-of evaluation |
| Stratified CV mixes years | CV does not simulate future deployment | Add chronological validation |
| Tuning and policy selection reuse development folds | Selected development scores may be optimistic | Use nested or separate validation in further work |
| Some test information was inspected | Holdout independence is incomplete | Reserve a new cohort for confirmation |
| DTI 999 is treated as missing | Its meaning is undocumented | Verify the code and test alternative treatments |
| Income is self-reported | Reporting errors can affect predictions | Add verification information |
| Few observations have FICO below 660 | Evidence for this group is limited | Obtain more representative observations |
| Probabilities underestimate 2016 defaults | Scores understate later-cohort risk | Assess recalibration on recent labelled data |
| Protected-group labels are absent | Direct racial disparity assessment is unavailable | Obtain an appropriate subgroup evaluation dataset |
| Financial returns and losses are absent | Default reductions cannot be translated directly into profit | Add costs, exposure and recovery information |

## 16. Repository and Reproduction

### Files

| File | Purpose |
|---|---|
| [01_split_and_sample.ipynb](notebooks/01_split_and_sample.ipynb) | Temporal split and sampling |
| [02_eda.ipynb](notebooks/02_eda.ipynb) | Training-data exploration |
| [03_pipeline.ipynb](notebooks/03_pipeline.ipynb) | Pipeline explanation and checks |
| [preprocessing.py](notebooks/preprocessing.py) | Shared preprocessing |
| [04_modeling_baseline.ipynb](notebooks/04_modeling_baseline.ipynb) | Baseline, CV, importance and error analysis |
| [05_tuning_and_final_evaluation.ipynb](notebooks/05_tuning_and_final_evaluation.ipynb) | Tuning, ensembles, policies and final evaluation |
| [requirements.txt](requirements.txt) | Dependency list |
| `data/train.csv` | 150,000-loan development sample |
| `data/test.csv` | 50,000-loan test sample |

### Setup

```bash
git clone https://github.com/o-tunamonana/lending-club-default-prediction.git
cd lending-club-default-prediction
python -m pip install -r requirements.txt
python -m pip install "scikit-learn>=1.4" scipy joblib threadpoolctl
cd notebooks
jupyter notebook
```

Notebook 05 searches `HistGradientBoostingClassifier.max_features`, requiring scikit-learn 1.4 or later.

### Execution

Run notebooks from the `notebooks/` directory in numerical order.

To regenerate the samples, download the raw CSV from Zenodo and place it at `data/LC_loans_granting_model_dataset.csv`. Existing sample files allow notebooks 02–05 to run without regenerating the split.

Notebook 04 currently reads `train.csv` from its working directory. When running from `notebooks/`, set its loading line to:

```python
X_train, y_train = load_split("../data/train.csv")
```

Notebook 05 saves intermediate results under `notebooks/results/` and figures under `notebooks/figures/`. Remove the relevant caches before recomputing after changes to code, data or parameters.

Sampling, shuffled CV, randomized search, and approval tie-breaking use seed **42**.

## 17. Conclusion

Application-time features improve risk ranking beyond FICO alone.

At **80% approval**, tuned gradient boosting retained 40,000 test loans with a **19.50%** default rate, compared with **21.56%** for FICO. This represents **827 fewer defaulted loans** at the same approval volume.

The main remaining issue is calibration: ranking transferred to 2016, but probabilities underestimated its higher default level. Further work should assess chronological validation, recalibration, and dependence on geographic features.

## References

- [Lending Club loan dataset for granting models, version 0.1](https://doi.org/10.5281/zenodo.11295916)
