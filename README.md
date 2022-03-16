# Deploying a Scalable ML Pipeline in Production

This project uses Census Income Data to build and evaluate the binary classification model (determine if a person makes over 50K a year).
- Git and Data Version Control (open-source version control system for ML projects) are used to track code, data and model.  
- After creating the model, its performance is evaluated on slices of the data. 
- A model card outlines the creation, use and shortcomings of the current model. 
- Continuous Integration and Continuous Deployment framework is used to ensure pipeline passes a series of unit tests before deployment. 
- An API will be written using FastAPI and tested locally. 

## Executing the pipeline

### Data Cleaning 

```
python main.py --mlstep basic_clean
```

### Model training and testing

```
python main.py --mlstep model_functions
```

### Model score Check score on latest saved model 
```
python main.py --mlstep slice_performance
```

### Run entire pipeline 
To run the entire pipeline in sequence, use 
```
python main.py --mlstep all
```

### FastAPI Testing
For local testing, execute:
```
uvicorn app_server:app --reload
```

## Continuous Integration and Continuous Deployment
Using Github actions, every new commit triggers a [test pipeline](https://github.com/wink-thedeeplearner/ml_dvc_heroku_fastapi/blob/main/.github/workflows/app_test.yml). 
- This workflow triggers a DVC pull and exectutes Pytest and Flake8 with Github actions.