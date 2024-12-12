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

##Set a runtime timer
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


# Create new dataframe for features variables or training columns for supervised learning
training_features = ["snr", "magnitude_mean", "magnitude_std", "magnitude_skew", "magnitude_kurtosis", "phase_mean", "phase_std", "phase_skew", "phase_kurtosis", "spectral_entropy", "peak_frequency", "average_power"]
feature_transform = features_df[training_features]
X = pd.DataFrame(columns=training_features, data=feature_transform, index=features_df.index)

##Set a runtime timer for training and prediction
start_time = time.time()
# split the data for 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# Decision tree classifier
tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

# prediction on train set
y_pred_train = tree_model.predict(X_train)


# Predictions on the  dataset
y_pred_test = tree_model.predict(X_test)

# Evaluate the model accurecy
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the model: {elapsed_time:.3f} seconds")


#Hyperparmater tuning using GridSearchCV
#param_grid = {
#    'max_depth': range(1, 10, 1),
#    'min_samples_leaf': range(1, 20, 2),
#    'min_samples_split': range(2, 20, 2),
#    'criterion': ["entropy", "gini"]
#}

param_grid = dict(
    max_depth= [10, 20, 30, None],
    min_samples_leaf= [2, 5, 10],
    min_samples_split= [1, 2, 4],
    criterion= ["entropy", "gini"]
)

# GridSearchCV
grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_tree_model = grid_search.best_estimator_ # Get the best estimator from the grid search ## update the tree_model accordingly with the best parameters and run the model again with the new paramters.
y_pred_test = tree_model.predict(X_test)
best_params = grid_search.best_params_
accuracy = accuracy_score(y_test, y_pred_test)

# Best score and estimator
print(f"Best parameters: {best_params}")
print("best accuracy", grid_search.best_score_)
print("best_tree_mode", grid_search.best_estimator_)

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
model_file_name = "Model3_ConfusionMatrix.png"
save_path = os.path.join(DIRECTORY, model_file_name)
plt.savefig(save_path)


# Print Classification Report
with open('Classification_report_Model_3.txt', 'w') as f:
    print("Classification Report for Modulation Types:", file=f)
    print("Train Result:n================================================", file=f)

    print(classification_report(y_train, y_pred_train, target_names=label_encoder.classes_), file=f)
    print("Test Result:n================================================", file=f)
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_), file=f)

#Accuracy vs SNR
unique_snrs = sorted(set(X_test['snr'])) # re-ordered SNR from min to max, without repeating
accuracy_per_snr = []
for snr in unique_snrs:
    snr_indices = np.where(X_test['snr'] == snr)
    X_snr = X_test[X_test['snr']==snr]
    y_snr=y_test[snr_indices]
    y_pred_test_snr= tree_model.predict(X_snr)
    accuracy = accuracy_score(y_snr, y_pred_test_snr)
    accuracy_per_snr.append(accuracy * 100)  # Convert to percentage
    print(f"SNR: {snr} dB, Accuracy: {accuracy * 100:.2f}%")

#Scale the SNR values between -18 and 20 using normalization
SNR_min = -20
SNR_max = 18
scaled_SNR = [(x - min(unique_snrs)) / (max(unique_snrs) - min(unique_snrs)) * (SNR_max - SNR_min) + SNR_min for x in unique_snrs]
# Plot Model Recognition Accuracy vs. SNR
plt.figure(figsize=(10, 6))
plt.plot(scaled_SNR, accuracy_per_snr, 'b-o', label='Model Accuracy')

# Find the maximum Accuracy value and its corresponding SNR
max_accuracy = max(accuracy_per_snr)
max_accuracy_index = accuracy_per_snr.index(max_accuracy)
max_accuracy_snr = scaled_SNR[max_accuracy_index]

plt.plot(max_accuracy_snr, max_accuracy, marker='o', markersize=8, color='r', label='_nolegend_')
# Annotate the max point on the plot
plt.annotate(f'Max: {round(max_accuracy,2)}%',
            xy=(max_accuracy_snr, max_accuracy),
            xytext=(max_accuracy_snr - 5, max_accuracy + 5),
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="white", linewidth=1, alpha=0.5),
            fontsize=8)

plt.xlabel("SNR (dB)")
plt.ylabel("Recognition Accuracy (%)")
plt.title("Recognition Accuracy vs. SNR for Modulation Classification")
plt.legend()
plt.grid(True)
plt.ylim(0, 100)

#Save file to desktop
DIRECTORY="."
model_file_name = "Model3_Accuracy_vs_SNR.png"
save_path = os.path.join(DIRECTORY, model_file_name)
plt.savefig(save_path)
plt.show(block=False)
