# Model Card: Census Income Classification Model

## Model Details
This model predicts whether an individual's income exceeds \$50K per year based on demographic and employment features from the U.S. Census Bureau dataset.  
It was developed as part of the **Deploying a Scalable ML Pipeline with FastAPI** project using a **Random Forest Classifier** implemented with **scikit-learn**.

- **Algorithm:** Random Forest Classifier  
- **Framework:** Scikit-learn  
- **Programming Language:** Python 3.10  
- **Environment:** Conda environment `fastapi`

## Intended Use
This model is intended for **educational and experimental use** to demonstrate:
- Building and training a machine learning pipeline  
- Evaluating model performance on data slices  
- Deploying a trained model via FastAPI  

It is **not intended for production use** or any real-world applications involving hiring, income prediction, or financial decisions.

## Training Data
- **Dataset:** `census.csv` (U.S. Census Bureau Adult Income dataset)  
- **Total Records:** 32,561 samples  
- **Features:** 14 features including `age`, `workclass`, `education`, `occupation`, `relationship`, `race`, `sex`, `hours-per-week`, and others.  
- **Target Variable:** `salary` (<=50K or >50K)

### Data Split
- **Training Data:** 80% (26,048 records)  
- **Testing Data:** 20% (6,513 records)

## Evaluation Data
The evaluation data consists of the 20% test split that was held out from training.  
No data leakage occurred between training and evaluation.

## Metrics
The model was evaluated on the test set using **Precision**, **Recall**, and **F1 Score**.  
Results are as follows:

| Metric | Value |
|--------|--------|
| Precision | 0.7419 |
| Recall | 0.6384 |
| F1 Score | 0.6863 |

### Performance by Data Slice
Performance was also computed on categorical slices to ensure fairness and consistency.  
A few examples from the slice report (`slice_output.txt`) are shown below:


## Ethical Considerations
This dataset includes sensitive demographic attributes such as race and gender.  
Any real-world deployment of this model may risk reflecting or amplifying societal biases present in the data.  
Therefore, it should not be used for real decision-making or employment screening.

## Caveats and Recommendations
- The model reflects patterns in historical U.S. census data and may not generalize to other countries or time periods.  
- Further tuning, feature scaling, or alternative algorithms may improve accuracy.  
- Bias and fairness audits should be conducted before any production use.  
- This project is for **educational demonstration only**.

---
