import time
from feature_extraction import features_extraction_
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


# open the input file
with open('RML2016.10a_dict.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # Use 'latin1' encoding to avoid Unicode errors

with open('data_details.txt', 'w') as f:
    for key, signals in data.items():
        mod_type, snr = key
        print(f"Modulation Type: {mod_type}, SNR: {snr}, Shape: {signals.shape}", file=f)

##Set a runtime timer for the training only
start_time = time.time()

#creat dataframe
features_df = features_extraction_(data)

elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the model: {elapsed_time:.3f} seconds")


# Create new dataframe for target variable or label column for supervised learning
y = pd.DataFrame(features_df['signal_type'])

# Label encoding the target variable
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)

# convert the target variable into into numerical valuesto a numerical value
#d = {'BPSK': 0, 'QPSK': 1, 'QAM16': 2, 'WBFM':3}
#features_df['signal_type'] = features_df['signal_type'].map(d)
#y=features_df['signal_type']



# Create new dataframe for features variables or training columns for supervised learning
training_features = ["magnitude_mean", "magnitude_std", "magnitude_skew", "magnitude_kurtosis", "phase_mean", "phase_std", "phase_skew", "phase_kurtosis", "spectral_entropy", "peak_frequency", "average_power"]
feature_transform = features_df[training_features]
X = pd.DataFrame(columns=training_features, data=feature_transform, index=features_df.index)

# split the data for 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# Decision tree classifier
tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

#Hyperparmater tuning using GridSearchCV
param_grid = {
    'max_depth': range(1, 10, 1),
    'min_samples_leaf': range(1, 20, 2),
    'min_samples_split': range(2, 20, 2),
    'criterion': ["entropy", "gini"]
}


# GridSearchCV
grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, 
                           cv=5, verbose=True)
grid_search.fit(X_train, y_train)

# Best score and estimator
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)

# Predictions on the  dataset
y_test = tree_model.predict(X_test)
y_pred_train = svm_model.predict(X_train)

# Evaluate the model accurecy
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix evaluation
confusionMatrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(confusionMatrix)

# heatmap plot for confusion matrix
plt.figure(figsize=(11, 11))
sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="Blues",
xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Signal Type Confusion Matrix", fontsize=14)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)

#Save file to desktop
DIRECTORY="."
model_file_name = "ConfusionMatrix.png"
save_path = os.path.join(DIRECTORY, model_file_name)
plt.savefig(save_path)

#Visualize the Decision tree
plt.figure(figsize=(18, 15))
plot_tree(tree_model, filled=True, feature_names=training_features,
          class_names=features_df['signal_type'], fontsize=5, label='root')
plt.savefig('tree_high_dpi', dpi=100)
plt.show()