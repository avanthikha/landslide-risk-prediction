# Landslide Risk Prediction

Out of 5,000 weather and terrain readings, only 12 were labelled "Very High" risk.
A lazy model could ignore them and still score 99% accuracy. Those 12 are the
ones that matter.

This project predicts landslide risk (Low, Moderate, High, Very High) from five
simple measurements: temperature, humidity, rainfall, soil moisture and elevation.
It also explains each prediction, so you can see *why* the model raised the alarm.

## How it works

The data is heavily skewed toward "Low", so the majority class is undersampled
and the rare classes are rebuilt with SMOTE-ENN until every risk level has a fair
voice. Ten classifiers are then trained and compared, from simple linear models
to gradient-boosted trees. The best one is examined with a confusion matrix,
ROC curves, and LIME explanations.

## What came out on top

Tree-based models won clearly. Random Forest and Gradient Boosting both reached
99.9% accuracy, with XGBoost close behind at 99.8%. Linear and neural models
struggled, and AdaBoost trailed at 66%. Full scores for all ten models are in
`results/`.


## Try it

```
pip install -r requirements.txt
python landslide_vit.py
```

---
Built by Avanthikha