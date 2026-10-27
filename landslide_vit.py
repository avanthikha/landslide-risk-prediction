#lib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


#%% load dataset
# Load and inspect data
data = pd.read_csv(r"data\regenerated_landslide_risk_dataset.csv")

#%%

#assign digits to label
enc= LabelEncoder()
data['Landslide Risk Prediction']= enc.fit_transform(data['Landslide Risk Prediction'])


#%% to know what label is assigned to alph

mapping = dict(zip(enc.classes_, range(len(enc.classes_))))
print("Category to Number Mapping:")
print(mapping)

#%%

class_counts = data['Landslide Risk Prediction'].value_counts()
print("Class counts:")
print(class_counts)


#%%

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
from imblearn.over_sampling import ADASYN

# Load features and target
X = data.drop(['Landslide Risk Prediction'], axis=1)
y = data['Landslide Risk Prediction']

# Define target samples per class
target_class_size = 1000
sampling_strategy_under = {1: target_class_size}  # Under-sample class 1 to 1000 samples

# 1. Apply undersampling for the majority class
undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=0)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# 2. Apply SMOTE to balance minority classes to 1000 samples each
sampling_strategy_over = {0: target_class_size, 2: target_class_size, 3: target_class_size}
adasyn = SMOTEENN(random_state=0, sampling_strategy=sampling_strategy_over)
X_balanced, y_balanced = adasyn.fit_resample(X_resampled, y_resampled)
# Check class distribution
print("Balanced class distribution:")
print(Counter(y_balanced))



#%% 

xtrain, xtest, ytrain, ytest = train_test_split(X_balanced,y_balanced, train_size=0.75, random_state=0)



#%%

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, jaccard_score, 
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier, BaggingClassifier
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.experimental import enable_hist_gradient_boosting
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


xtrain, xtest, ytrain, ytest = train_test_split(X_balanced, y_balanced, train_size=0.75, random_state=0)
# Define classifiers
classifiers = {
    'Support Vector Machine (Linear)': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(random_state=0),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'MLP': MLPClassifier(),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis()
    
}

results = []

# Train each classifier and evaluate
for name, clf in classifiers.items():
    clf.fit(xtrain, ytrain)  # Fit the classifier
    ypred = clf.predict(xtest)  # Make predictions

    # Calculate evaluation metrics
    accuracy = accuracy_score(ytest, ypred)
    precision = precision_score(ytest, ypred,average='weighted')
    recall = recall_score(ytest, ypred,average='weighted')
    
    f1 = f1_score(ytest, ypred,average='weighted')
    
    mcc = matthews_corrcoef(ytest, ypred)

    # Store the results
    results.append({
        'Classifier': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv(r"results\classifier_results_75_smote+ennsampling.csv", index=False)

# Print the results DataFrame
print(results_df)

#%% without SMOTEENN


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

xtrain, xtest, ytrain, ytest = train_test_split(X_resampled,y_resampled, train_size=0.75, random_state=89)


# Initialize Random Forest Classifier
model = RandomForestClassifier(random_state=99)

# Train the model on the training data
model.fit(xtrain, ytrain)

# Make predictions on the test set
ypred = model.predict(xtest)

# Calculate evaluation metrics
accuracy_rf = accuracy_score(ytest, ypred)
precision_rf = precision_score(ytest, ypred, average='weighted')
recall_rf = recall_score(ytest, ypred, average='weighted')
f1_rf = f1_score(ytest, ypred, average='weighted')
mcc_rf = matthews_corrcoef(ytest, ypred)

# Print the final evaluation metrics
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc_rf:.4f}")

#%%


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

xtrain, xtest, ytrain, ytest = train_test_split(X_balanced, y_balanced, train_size=0.75, random_state=0)

# Initialize Random Forest Classifier
model = RandomForestClassifier(random_state=99)

# Train the model on the training data
model.fit(xtrain, ytrain)

# Make predictions on the test set
ypred = model.predict(xtest)

# Calculate evaluation metrics
accuracy_rf = accuracy_score(ytest, ypred)
precision_rf = precision_score(ytest, ypred, average='weighted')
recall_rf = recall_score(ytest, ypred, average='weighted')
f1_rf = f1_score(ytest, ypred, average='weighted')
mcc_rf = matthews_corrcoef(ytest, ypred)

# Print the final evaluation metrics
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc_rf:.4f}")

#%% confusion matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
)
cm = confusion_matrix(ytest, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Random Forest")
plt.show()

#%% receiver operating charactstics

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

# Binarize the labels
classes = np.unique(ytest)
ytest_bin = label_binarize(ytest, classes=classes)
yprob = model.predict_proba(xtest)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i, cls in enumerate(classes):
    fpr[cls], tpr[cls], _ = roc_curve(ytest_bin[:, i], yprob[:, i])
    roc_auc[cls] = auc(fpr[cls], tpr[cls])

# Plot all ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
for cls, color in zip(classes, colors):
    plt.plot(fpr[cls], tpr[cls], color=color,
             label=f"Class {cls} (AUC = {roc_auc[cls]:.4f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.show()

#%% LIME

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np

# ✅ Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(xtrain),
    feature_names=xtrain.columns,
    class_names=np.unique(ytrain).astype(str),
    mode='classification'
)

# ✅ Select a test instance (you can change index i)
i = 2

# Get model prediction for that instance
#pred_proba = model.predict_proba([xtest.iloc[[i]]])[0]
pred_proba = model.predict_proba(xtest.iloc[[i]])[0]

pred_class_index = np.argmax(pred_proba)
pred_class = model.classes_[pred_class_index]

print(f"\n🧠 Test Instance: {i}")
print(f"Predicted Class: {pred_class}")
print(f"Class Probabilities: {dict(zip(model.classes_, np.round(pred_proba, 4)))}")

# ✅ Generate LIME explanation
exp = explainer.explain_instance(
    data_row=xtest.iloc[i],
    predict_fn=model.predict_proba
)

# ✅ Display chart
fig = exp.as_pyplot_figure()
plt.title(f"LIME Explanation for Test Instance {i}\nPredicted Class: {pred_class}", fontsize=12)
plt.show()

# Optional: Save the chart
# fig.savefig(f"lime_explanation_instance_{i}_{pred_class}.png", dpi=300, bbox_inches='tight')





