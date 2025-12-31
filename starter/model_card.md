# Model Card: Income Classification Model

## Model Details
This repository contains a binary classification model that predicts whether an individual's annual income is greater than 50K based on tabular demographic and employment features. The model implementation is in starter/ml/model.py and uses scikit-learn for training and inference. The model is packaged in the starter Python package.

## Intended Use
The model is intended to assist in exploratory analyses and educational tasks for predicting whether an individual's income exceeds 50K given features such as age, workclass, education, marital-status, occupation, relationship, race, sex, native-country, hours-per-week, and capital-gain. It is not intended for automated high-stakes decisions affecting individuals (for example, hiring, credit, insurance underwriting, or legal determinations) without additional human oversight and fairness audits.

## Training Data
The model is trained on a cleaned census-style dataset (commonly the UCI "Adult" / Census Income dataset or an equivalent cleaned CSV present in this repository). Training features include categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country) and numerical features (age, hours-per-week, capital-gain, etc.). Please see starter/data or the preprocessing code in starter/ml/data.py for details about the exact training file and preprocessing pipeline.

## Evaluation Data
Evaluation is performed on a held-out portion of the cleaned dataset or on a dedicated test split produced during preprocessing. The evaluation uses the same preprocessing pipeline as training to ensure consistency. The repository also includes a slice performance function to measure model performance per subgroup (see starter/ml/model.py slice_performance).

## Metrics
The following metrics are used to evaluate model performance:
- Precision: proportion of positive predictions that are correct.
- Recall: proportion of true positives that are identified.
- F-beta (beta=1): harmonic mean of precision and recall (F1 score).

Example results from a representative run :
- Precision: 0.74
- Recall: 0.63
- F1 (fbeta): 0.68

## Ethical Considerations
The dataset contains sensitive attributes such as race, sex, and native-country, which can introduce biases into predictions. Performance can vary across demographic subgroups. Users should evaluate subgroup performance (for example using slice_performance) and consider fairness mitigation strategies if biased behavior is detected. The model should not be used as the sole basis for decisions that materially affect individuals.

## Caveats and Recommendations
- The model's performance depends on the training data distribution; performance may degrade under distribution shifts.
- Evaluate performance on representative, up-to-date data before deployment.
- Perform fairness audits across protected groups and consider threshold adjustments, reweighing, or post-processing mitigation if disparities appear.
- Implement monitoring to detect data drift and retrain periodically.
- Use human-in-the-loop review for high-impact decisions.
