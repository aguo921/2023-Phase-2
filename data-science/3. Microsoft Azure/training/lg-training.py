# Import libraries
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

# Get the experiment run context
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args = parser.parse_args()
reg = args.reg

# load the market segmentation dataset
print("Loading Data...")
market_segmentation = pd.read_csv('market_segmentation_interaction.csv')

# Separate features and labels
X, y = market_segmentation.drop(columns="Segmentation"), market_segmentation.Segmentation

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LogisticRegression(C=1/reg).fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_scores = model.predict_proba(X_test)
for class_of_interest in ["A", "B", "C", "D"]:
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
    auc = roc_auc_score(y_onehot_test[:,class_id],y_scores[:,class_id])
    print(f'AUC {class_of_interest} vs rest: ' + str(auc))
    run.log(f'AUC {class_of_interest} vs rest', np.float(auc))

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/lg-model.pkl')

run.complete()
