
# Import libraries
import argparse, joblib, os
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

# Get the experiment run context
run = Run.get_context()

# Get script arguments
parser = argparse.ArgumentParser()

# Set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--max_features', type=int, dest='max_features', default=4)
parser.add_argument('--max_depth', type=int, dest='max_depth', default=None)
parser.add_argument('--min_samples_split', type=int, dest='min_samples_split', default=2)
parser.add_argument('--min_samples_leaf', type=int, dest='min_samples_leaf', default=1)
args = parser.parse_args()
max_features = args.max_features
max_depth = args.max_depth
min_samples_split = args.min_samples_split
min_samples_leaf = args.min_samples_leaf

# Log Hyperparameter values
run.log('Maximum depth of tree', max_depth)
run.log('Maximum features per split', max_features)
run.log('Minimum samples requierd for split', min_samples_split)
run.log('Minimum samples per leaf', min_samples_leaf)

# load the market segmentation dataset
print("Loading Data...")
market_segmentation = pd.read_csv('market_segmentation.csv')

# Separate features and labels
X, y = market_segmentation.drop(columns="Segmentation"), market_segmentation.Segmentation

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a logistic regression model
print(
    'Training a random forest model with max features of', max_features,
    'max depth of', max_depth,
    'min samples split of', min_samples_split,
    'and min_samples_leaf of', min_samples_leaf
)
model = RandomForestClassifier(
    max_features=max_features,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf
).fit(X_train, y_train)

# Calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# Calculate AUC
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_scores = model.predict_proba(X_test)
for class_of_interest in ["A", "B", "C", "D"]:
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
    auc = roc_auc_score(y_onehot_test[:,class_id],y_scores[:,class_id])
    print(f'AUC {class_of_interest} vs rest: ' + str(auc))
    run.log(f'AUC {class_of_interest} vs rest', np.float(auc))

# Save the model in the run outputs
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/rf-model.pkl')

run.complete()
