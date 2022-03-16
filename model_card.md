# Model Card

For additional information, please reference the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Khai Win created the model. This model uses:
- Gradient Boosting Classifier 
- the default hyperparameters in scikit-learn

## Intended Use

The model will be used to predict if a person's annual income is over 50K based on attributes from 1994 Census Data.

## Data
The dataset used in this project can be found on UCI Machine Learning Repo:
https://archive.ics.uci.edu/ml/datasets/census+income ; 

- Training Data: training is done using 80% of this data.

- Evaluation Data: evaluation is done using the remaining 20% of this data.

## Metrics

The precision, recall, and F1 score metrics were used to evaluate the model.

## Ethical Considerations

Proper care should be taken when using this model since he dataset used for this model contains sensitive demographic attributes, such as gender, race, and origin country. 
- The outcome of the model has not been rigorously assessed for bias.
- The model may potentially discriminate people based on the demographic attributes used.
Hence, further investigation should be done before using the model.

## Caveats and Recommendations

- Since the goal of this project is to deploy ML models using FastAPI, only basic cleaning/preprocessing was performed. 
To improve the results, future work should consider using additional techniques on feature selection and engineering. 

- The gender attribute used in the project is binary: male or not male(female). Future work should include a broad spectrum of gender classes.


