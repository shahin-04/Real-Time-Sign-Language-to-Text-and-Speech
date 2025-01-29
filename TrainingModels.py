import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tabulate import tabulate

# Language mappings from createdataset.py
LANGUAGES = {
    'ISL': 'ISL_data.pickle',
    'ASL': 'ASL_data.pickle',
    'BANZSL': 'BANZSL_data.pickle',
    'DGS': 'DGS_data.pickle',
    'LSF': 'LSF_data.pickle',
    'ArSL': 'ArSL_data.pickle'
}

# Define machine learning models
MODELS = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Store results and best models
all_results = {}
best_models = {}

for lang, data_file in LANGUAGES.items():
    print(f"\nTraining models for {lang} dataset...")
    try:
        # Load dataset
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])

        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Initialize language-specific results
        all_results[lang] = {}
        best_accuracy = 0
        best_model_name = None
        best_model = None

        # Train models
        for model_name, model in MODELS.items():
            print(f"  Training {model_name}...")
            model.fit(x_train, y_train)

            # Evaluate model
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_accuracy = accuracy_score(y_train, y_train_pred) * 100
            test_accuracy = accuracy_score(y_test, y_test_pred) * 100

            # Save results
            all_results[lang][model_name] = {
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy
            }
            print(f"    {model_name}: Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%")

            # Update best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_name = model_name
                best_model = model

        # Save the best model for the language
        best_models[lang] = {'model_name': best_model_name, 'model': best_model}
        with open(f'{lang}_best_model.p', 'wb') as f:
            pickle.dump({'model': best_model}, f)

        print(f"Best model for {lang}: {best_model_name} with Test Accuracy = {best_accuracy:.2f}%")
        print(f"Saved best model for {lang} as {lang}_best_model.p")

    except FileNotFoundError:
        print(f"Error: Dataset file {data_file} not found. Skipping {lang}.")

# Generate tabular and graphical comparisons
# Detailed table for training and testing accuracy
detailed_table = [["Language", "Random Forest", "SVM", "KNN", "Logistic Regression", "Decision Tree"]]
for lang, results in all_results.items():
    row_train = [lang + " (Train)"]
    row_test = [lang + " (Test)"]
    for model in MODELS.keys():
        row_train.append(f"{results[model]['Train Accuracy']:.2f}")
        row_test.append(f"{results[model]['Test Accuracy']:.2f}")
    detailed_table.append(row_train)
    detailed_table.append(row_test)

print("\nDetailed Accuracy Table (Training and Testing):")
print(tabulate(detailed_table, headers="firstrow", tablefmt="grid"))

# Testing accuracy comparison
testing_table = [["Language"] + list(MODELS.keys())]
for lang, results in all_results.items():
    row = [lang]
    for model in MODELS.keys():
        row.append(f"{results[model]['Test Accuracy']:.2f}")
    testing_table.append(row)

print("\nTesting Accuracy Comparison Table:")
print(tabulate(testing_table, headers="firstrow", tablefmt="grid"))

# Graphical representation of testing accuracy
for lang, results in all_results.items():
    model_names = list(MODELS.keys())
    test_accuracies = [results[model]['Test Accuracy'] for model in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, test_accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'])

    # Add accuracy values above bars
    for bar, accuracy in zip(bars, test_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{accuracy:.2f}%", ha='center', va='bottom', fontsize=10, color='black')

    # Title and labels
    plt.title(f"Testing Accuracy Comparison for {lang}", fontsize=14, pad=20)
    plt.xlabel("Models", fontsize=13)
    plt.ylabel("Testing Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{lang}_testing_accuracy_comparison.png")
    plt.show()
