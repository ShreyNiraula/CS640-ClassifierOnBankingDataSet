from sklearn.svm import SVC
import logging
from sklearn.linear_model import LogisticRegression
import urllib.request
import zipfile
import sys
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.neural_network import MLPClassifier
from itertools import product
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import warnings
from pandas.errors import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

# Preprocessor class 
class CustomPreprocessor:
    def __init__(self, is_dropping, is_pca, is_SMOTE, is_chi_square_dropping, is_pdays_binary_conveted, is_duration_0_removed,
                 is_skewness_handled,
                 is_campaign_previous_duration_capped, pca_threshold, top_k, p_value, corr_threshold):
        self.is_dropping = is_dropping
        self.is_pca = is_pca
        self.is_SMOTE = is_SMOTE
        self.is_chi_square_dropping = is_chi_square_dropping
        self.is_pdays_binary_conveted = is_pdays_binary_conveted
        self.pca_threshold = pca_threshold
        self.top_k = top_k
        self.p_value = p_value
        self.is_duration_0_removed = is_duration_0_removed
        self.is_campaign_previous_duration_capped = is_campaign_previous_duration_capped
        self.is_skewness_handled = is_skewness_handled
        self.corr_threshold = corr_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.cap_values = {}
        self.skewness_transformations = {}
        self.drop_cols = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.top_features = []
        self.smote = None
        self.rare_categories = {}
        self.target_encoding_mappings = {}
        self.onehot_encoder_obj = OneHotEncoder(handle_unknown="ignore")
        self.correlation_drop_cols = []
        self.chi_square_drop_list = []
        self.succeeded_dropping = False
        self.succeeded_chisquare = False
        self.succeeded_handle_duration_0_removed = False
        self.succeeded_cap_conversion = False
        self.succeeded_pdays_conversion = False
        self.succeeded_skewness_conversion = False
        self.succeeded_encoding = False
        self.succeeded_scaling = False
        self.succeeded_pca = False
        self.succeeded_encoding = False
        self.succeed_dropping = False

    def correlation_analysis(self, X, y):
        corr_matrix = X[self.numeric_cols].corr()

        # plot the heatmap:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Feature Correlation Heatmap")
        # plt.show()

        highly_correlated_cols = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.corr_threshold:
                    colname_i = corr_matrix.columns[i]
                    colname_j = corr_matrix.columns[j]
                    highly_correlated_cols.add(colname_i)
                    highly_correlated_cols.add(colname_j)

        # print(f"Highly correlated features: {highly_correlated_cols}")

        # Retaining one from highly_correlated_cols which is most correlated to y, and adding other cols to drop_cols list
        corr_with_target = X[list(highly_correlated_cols)].corrwith(y)
        max_corr_col = abs(corr_with_target).idxmax()

        for col in highly_correlated_cols:
            if col != max_corr_col:
                self.correlation_drop_cols.append(col)
        # print(f"These are removed from the list of correlated cols: {self.correlation_drop_cols}")

    def pdays_to_binary_fit(self, X):
        if self.is_pdays_binary_conveted:
            X["contacted_before"] = np.where(X["pdays"] == 999, 0, 1)
            self.drop_cols.append("pdays")
            self.succeeded_pdays_conversion = True
        return X

    def pdays_to_binary_transform(self, X):
        if self.is_pdays_binary_conveted:
            if not self.succeeded_pdays_conversion:  # if val/test, do pdays_to_binary_fit, for train skip
                self.pdays_to_binary_fit(X)
        self.succeeded_pdays_conversion = False
        return X

    def handle_skewness_fit(self, X):
        if self.is_skewness_handled:
            for col in self.numeric_cols:
                skew_val = X[col].skew()
                if skew_val > 2:
                    self.skewness_transformations[col] = "log"
                    X[col] = np.log1p(X[col])
                elif 1 < skew_val <= 2:
                    self.skewness_transformations[col] = "sqrt"
                    X[col] = np.sqrt(X[col])
                elif skew_val < -2:
                    self.skewness_transformations[col] = "inverse"
                    X[col] = 1 / (X[col] + 1e-6)
                # elif -2 <= skew_val < -1:
                #     self.skewness_transformations[col] = "square"
                #     X[col] = (X[col])**2
                else:
                    self.skewness_transformations[col] = "none"
            self.succeeded_skewness_conversion = True
        return X

    def handle_skewness_transform(self, X):
        if self.is_skewness_handled:
            if not self.succeeded_skewness_conversion:  # for val and test, use skewness_transformations to do exact same
                for col in self.numeric_cols:
                    if self.skewness_transformations[col] == "log":
                        X[col] = np.log1p(X[col])
                    elif self.skewness_transformations[col] == "sqrt":
                        X[col] = np.sqrt(X[col])
                    elif self.skewness_transformations[col] == "inverse":
                        X[col] = 1 / (X[col] + 1e-6)
                    # elif self.skewness_transformations[col] == "square":
                    #     X[col] = (X[col])**2
            self.succeeded_skewness_conversion = False
        return X

    def encode_features_fit(self, X):
        X_backup = X[self.numeric_cols].copy()
        self.rare_categories = {}
        self.cols_to_target_encode = []
        self.cols_to_onehot_encode = []

        for col in self.categorical_cols:
            value_counts = X[col].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < 0.05].index
            self.rare_categories[col] = rare_categories
            X.loc[:, col] = X[col].replace(rare_categories, 'Others')

            if X[col].nunique() > 10:
                self.cols_to_target_encode.append(col)
                self.target_encoding_mappings[col] = X.groupby(col)[y.name].mean()
                X[col] = X[col].map(self.target_encoding_mappings[col])
            else:
                self.cols_to_onehot_encode.append(col)

        self.onehot_encoder_obj = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot_encoded = self.onehot_encoder_obj.fit_transform(X[self.cols_to_onehot_encode])
        onehot_feature_names = self.onehot_encoder_obj.get_feature_names_out(self.cols_to_onehot_encode)
        X = X.drop(columns=self.cols_to_onehot_encode, errors="ignore")

        X = pd.concat([X_backup, pd.DataFrame(onehot_encoded, columns=onehot_feature_names, index=X.index)],
                      axis=1)
        self.categorical_cols = onehot_feature_names

        self.succeeded_encoding = True

        return X

    def encode_features_transform(self, X):
        if not self.succeeded_encoding:  # Only process if encoding not already done
            categorical_features_val_test = X.select_dtypes(exclude=[np.number]).columns
            for col in categorical_features_val_test:
                # Apply rare category replacement
                if col in self.rare_categories and col in X.columns:
                    X[col] = X[col].replace(self.rare_categories[col], 'Others')

                # Apply target encoding for high-cardinality columns
                if col in self.target_encoding_mappings and col in X.columns:
                    X[col] = X[col].map(self.target_encoding_mappings[col])

            # Apply one-hot encoding for low-cardinality columns
            if self.onehot_encoder_obj:
                onehot_encoded = self.onehot_encoder_obj.transform(X[self.onehot_encoder_obj.feature_names_in_])
                onehot_feature_names = self.categorical_cols
                X = X.drop(columns=self.onehot_encoder_obj.feature_names_in_, errors="ignore")
                X = pd.concat(
                    [X, pd.DataFrame(onehot_encoded, columns=onehot_feature_names, index=X.index)], axis=1
                )

        self.succeeded_encoding = False  # Reset after processing for subsequent datasets
        return X

    def pca_fit(self, X):
        if self.is_pca:
            # X_scaled = self.scaler.transform(X)
            self.pca = PCA()
            self.pca.fit(X)
            cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
            n_components = (cumulative_variance >= self.pca_threshold).argmax() + 1

            # visualize
            # Plot the explained variance ratio to find the elbow point
            # plt.figure(figsize=(16, 12))
            # plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
            # plt.xlabel('Number of Components')
            # plt.ylabel('Cumulative Explained Variance')
            # plt.title('Elbow Plot for PCA')
            # plt.xticks(range(1, len(cumulative_variance) + 1))  # Ensure x-axis ticks align with components
            # plt.grid(True)
            # plt.show()

            self.pca = PCA(n_components=n_components).fit(X)
        return X

    def pca_transform(self, X):
        if self.is_pca:
            X_copy = X.copy()
            if self.pca:
                X_pca = self.pca.transform(X)
                X = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])], index=X_copy.index)
        return X

    def scalar_fit(self, X):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X=pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.succeed_scaling = True
        return X

    def scalar_transform(self, X):
        if not self.succeed_scaling:  # if not done previously,(for val/test)
            X_scaled = self.scaler.transform(X)
            X=pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.succeed_scaling = False
        return X

    def drop_cols_and_update_numeric_cols_fit(self, X):
        if self.is_dropping:
            self.drop_cols.extend(self.correlation_drop_cols)
            X.drop(self.drop_cols, axis=1, inplace=True)
            self.succeed_dropping = True

            # update
            self.numeric_cols = X.select_dtypes(include=[np.number]).columns
        return X

    def drop_cols_transform(self, X):
        if self.drop_cols:
            if not self.succeed_dropping:
                self.drop_cols.extend(self.correlation_drop_cols)
                X.drop(self.drop_cols, axis=1, inplace=True)
            self.succeed_dropping = False
            self.drop_cols = []
        return X

    def chi_square_pvalues_fit(self, X, y):
        if self.is_chi_square_dropping:
            chi2_scores, p_values = chi2(X[self.categorical_cols], y)
            chi2_results = pd.DataFrame(
                {'Feature': self.categorical_cols, 'Chi2 Score': chi2_scores, 'P-value': p_values})

            # Sort by Chi2 score, descending
            chi2_results = chi2_results.sort_values(by='Chi2 Score', ascending=False)

            # Apply p-value threshold for feature selection (e.g., p-value < 0.05)
            significant_features = chi2_results[chi2_results['P-value'] < self.p_value]['Feature'].tolist()

            retain_list = list(self.numeric_cols) + significant_features  # list to retain
            self.chi_square_drop_list = list(set(X.columns) - set(retain_list))  # list of cols to drop
            X.drop(self.chi_square_drop_list, axis=1, inplace=True)
            self.succeeded_chisquare = True
        return X

    def chi_square_pvalues_transform(self, X):
        if self.is_chi_square_dropping:
            if not self.succeeded_chisquare:  # for train, skip as cols are dropped in fit, but for val/test, drop here...
                X.drop(self.chi_square_drop_list, axis=1, inplace=True)
            self.succeeded_chisquare = False
        return X

    # only for Train set. Only the fit version
    def apply_smote(self, X, y):
        if self.is_SMOTE:
            if self.smote is None:
                self.smote = SMOTE(random_state=42)

            # Convert DataFrame to numpy array for SMOTE if necessary
            if isinstance(X, pd.DataFrame):
                X_np = X.values  # Convert to numpy array for SMOTE
            else:
                X_np = X

            X_resampled, y_resampled = self.smote.fit_resample(X_np, y.values if isinstance(y, pd.Series) else y)
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name, index=None)

            X = X_resampled
            y = y_resampled
        return X, y

    def box_plot_numerical_columns(self, X, y, list):
        for col in list:
            plt.figure(figsize=(6, 3))
            sns.boxplot(x='y', y=col, data=pd.concat([X, y], axis=1))
            plt.title(f'Box Plot of {col} for Each Class')
            # plt.show()

            for class_label in y.unique():
                class_data = X[y == class_label][col]
                median = np.median(class_data)
                q1 = np.percentile(class_data, 25)
                q3 = np.percentile(class_data, 75)
                min_val = np.min(class_data)
                max_val = np.max(class_data)
                iqr = q3 - q1
                lower_whisker = q1 - 1.5 * iqr
                upper_whisker = q3 + 1.5 * iqr

                # Print the statistics with two decimal places
            #     print(f"Class {class_label} - {col}:")
            #     print(f"  Min: {min_val:.2f}")
            #     print(f"  Lower Whisker: {max(min_val, lower_whisker):.2f}")  # Consider min value as well
            #     print(f"  Q1: {q1:.2f}")
            #     print(f"  Median: {median:.2f}")
            #     print(f"  Q3: {q3:.2f}")
            #     print(f"  Upper Whisker: {min(max_val, upper_whisker):.2f}")  # Consider max value as well
            #     print(f"  Max: {max_val:.2f}")
            # print("-" * 200)
            # print("\n")

    def histogram_numerical_columns(self, X, y, list):
        for col in list:
            plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization

            for class_label in y.unique():
                sns.histplot(X[y == class_label][col], label=f'Class {class_label}', kde=False)

            plt.title(f'Histogram of {col} for each class')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.legend()
            # plt.show()

            # for class_label in y.unique():
            #     value_counts = X[y == class_label][col].value_counts()
            #     # print(f"\nClass {class_label} - Distinct Values and Counts for {col}:", value_counts)

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns

        self.correlation_analysis(X, y)

        ###################### Turn on this to visualize the plots ##################
        # self.box_plot_numerical_columns(X, y, list=self.numeric_cols)
        # self.histogram_numerical_columns(X, y, list=self.numeric_cols)
        # X, y = self.handle_duration0_fit(X, y)
        ############################################################################

        X = self.pdays_to_binary_fit(X)

        X = self.drop_cols_and_update_numeric_cols_fit(X)

        X = self.handle_skewness_fit(X)
        X = self.encode_features_fit(X)

        X = self.chi_square_pvalues_fit(X, y)

        X = self.scalar_fit(X)  # scale anyhow, to maintain consistency. Else, PCA will scale, and if no PCA, unscaled data will be passed to SMOTE
        X = self.pca_fit(X)  # return df
        X, y = self.apply_smote(X, y)  # returns df

        return X, y

    def transform(self, X):
        X = self.pdays_to_binary_transform(X)
        X = self.drop_cols_transform(X)
        X = self.handle_skewness_transform(X)
        X = self.encode_features_transform(X)
        X = self.chi_square_pvalues_transform(X)
        X = self.scalar_transform(X)
        X = self.pca_transform(X)
        return X

    def fit_transform(self, X, y=None):
        X, y = self.fit(X, y)
        return self.transform(X), y


def load_data(url, filename):
    urllib.request.urlretrieve(url, filename)
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(".")
    return pd.read_csv("bank-additional/bank-additional-full.csv", sep=";")


def visualize_results(results, setting_options):
    varying_settings = {setting for setting, values in setting_options.items() if len(values) > 1}

    for model_name, model_results in results.items():
        setting_descriptions = []
        f1_scores = []
        recalls = []

        for settings, result in model_results.items():
            # Ensure `settings` is a dictionary by parsing the string
            import ast
            if isinstance(settings, str):
                settings = ast.literal_eval(settings)

            # Filter only the varying settings
            filtered_settings = {k: v for k, v in settings.items() if k in varying_settings}

            # Generate description and append metrics
            description = "\n".join(f"{k}={v}" for k, v in filtered_settings.items())
            setting_descriptions.append(description)
            f1_scores.append(result["F1 Score"])
            recalls.append(result.get("Recall", 0))  # Handle missing Recall gracefully

        # Create plot
        plt.figure(figsize=(16, 10))
        sns.barplot(x=f1_scores, y=setting_descriptions, orient="h", palette="viridis")
        plt.title(f"F1 Score and Recall for Different setting Combinations - {model_name}")
        plt.xlabel("F1 Score")
        plt.ylabel("Settings")

        # Add F1 and Recall values on bars
        for i, (f1, recall) in enumerate(zip(f1_scores, recalls)):
            plt.text(f1 + 0.01, i, f"F1: {f1:.5f}\nRecall: {recall:.5f}", color="red", va="center")

        plt.tight_layout()
        plt.show()


### PROGRAM FLOW BEGINS ###
# 1. Load and Split Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
filename = "bank-additional.zip"
df = load_data(url, filename)

X = df.drop("y", axis=1)
y = df["y"].map({"yes": 1, "no": 0})  # yes to 1, no to 0

# split to test and non-test
X_non_test, X_test, y_non_test, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Split Non-Test: Train and Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_non_test, y_non_test, test_size=0.2, random_state=42
)

# 2. Define models
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {"penalty": ["l2", "l1"], "C": [0.1, 1], "solver": ["saga"], "class_weight": [None]},
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "param_grid": {"max_depth": [5, 10], "min_samples_split": [2, 5], "class_weight": [None]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "param_grid": {"n_estimators": [100, 200], "max_depth": [5, 10], "class_weight": [None]}
    },
    "MLP": {
        "model": MLPClassifier(max_iter=1000, random_state=42),
        "param_grid": {"hidden_layer_sizes": [(50,), (100,)], "activation": ["relu", "tanh"]}
    },
    "SVM": {
        "model": SVC(random_state=42),
        "param_grid": {"C": [0.1, 1, 10], "kernel": ["rbf"], "class_weight": [None]}
    }
}

# 3. Define the base settings
baseline_setting = {
    "PCA": False,
    "SMOTE": False,
    "ChiSquare": False,
    "p_value": 0.05
}
setting_options = {
    "PCA": [False, True],
    "SMOTE": [True],
    "ChiSquare": [False, True],
    "p_value": [0.02, 0.05]
}


# initial setting combinations
initial_combinations = [
    dict(zip(setting_options.keys(), values)) for values in product(*setting_options.values())
]

# Filter combinations
filtered_combinations = []
seen_combinations = set()

for combination in initial_combinations:
    # If ChiSquare is False, ignore p_value and only keep unique combinations
    if not combination["ChiSquare"]:
        # Create a reduced representation ignoring p_value
        reduced_combination = combination.copy()
        reduced_combination.pop('p_value')  # Ignore p_value when ChiSquare is False
        # Add to filtered combinations if not already seen
        if tuple(reduced_combination.items()) not in seen_combinations:
            seen_combinations.add(tuple(reduced_combination.items()))
            filtered_combinations.append(combination)
    else:
        # Include combinations where ChiSquare is True
        filtered_combinations.append(combination)

setting_combinations = [baseline_setting] + filtered_combinations if len(baseline_setting) > 0 else filtered_combinations
print(f"length of setting_combinations: {len(setting_combinations)}")

# Initialize variables to track the best model
best_model_name = None
best_settings = None
best_f1 = 0
best_recall = 0
best_model = None
best_params = None
results = {}

# Set up logging to output to both console and a file
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"model_training_results_{current_time}.log"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler(output_file),  # Save to file
                        logging.StreamHandler(sys.stdout)  # Print to console
                    ])


logging.info(f"Model Training Results - {current_time}")
logging.info(f"Running {', '.join(models.keys())} under {len(setting_combinations)} settings:")

# Loop through models and settings
overall_start_time = time.perf_counter()
for model_name, model_info in models.items():
    results[model_name] = {}
    for ind, settings in enumerate(setting_combinations, start=1):
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a readable string for settings
        settings_str = ", ".join([f"{key}: {value}" for key, value in settings.items()])
        p_value_ = settings.get('p_value', None)

        if settings.get("ChiSquare", False):
            settings_str = ", ".join(
                f"{key}: {value}" + (
                    f" with top {key} selection: {p_value_} features" if key == "ChiSquare" and value else "")
                for key, value in settings.items()
            )
        else:
            settings_str = ", ".join(
                f"{key}: {value}" for key, value in settings.items()
            )

        if ind == 1:
            logging.info(f"\n{'=' * 30} BASELINE {model_name} {'=' * 70}")
        else:
            logging.info(f"\n{'=' * 150}")
        logging.info(f"Start time:{timestamp}")

        if ind == 1:
            logging.info(f"Running BASELINE {model_name} with {ind}th setting: {settings_str}")
        else:
            logging.info(f"Running {model_name} with {ind}th setting: {settings_str}")
        start_time = time.perf_counter()

        #4. Initialize custom preprocessor with settings
        preprocessor = CustomPreprocessor(
            is_pca=settings.get("PCA", False),
            is_SMOTE=settings.get("SMOTE", True),
            is_dropping=settings.get("Dropped", True),
            is_chi_square_dropping=settings.get("ChiSquare", False),
            is_pdays_binary_conveted=settings.get("PDAYSBinary", True),
            is_duration_0_removed=settings.get("is_duration_0_removed", False),
            is_campaign_previous_duration_capped=settings.get("Capped", False),
            is_skewness_handled=settings.get("HandleSkewness", True),
            pca_threshold=settings.get("pca_threshold", 0.96),
            top_k=settings.get("top_k", 15),
            p_value=settings.get("p_value", 0.05),
            corr_threshold=settings.get("corr_threshold", 0.9)
        )

        # Preprocess the training and validation data
        X_train_processed, y_train_processed = preprocessor.fit_transform(X_train.copy(), y_train.copy())
        X_val_processed = preprocessor.transform(X_val.copy())

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(model_info["model"], model_info["param_grid"], scoring="f1", cv=5, n_jobs=-1)
        grid_search.fit(X_train_processed, y_train_processed)

        # Predict on validation set and calculate metrics
        y_val_pred = grid_search.best_estimator_.predict(X_val_processed)
        f1 = f1_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        class_report = classification_report(y_val, y_val_pred, digits=5)

        # Neatly print the classification report
        logging.info(f"{'-' * 150}")
        logging.info(f"Classification Report for {model_name} with {ind}th setting: {settings_str}")
        logging.info(f"{'=' * 150}")
        logging.info(class_report)
        logging.info(f"{'=' * 150}")

        # Save results
        results[model_name][str(settings)] = {
            "F1 Score": f1,
            "Recall": recall,
            "Best Params": grid_search.best_params_,
            "Validation Report": classification_report(y_val, y_val_pred, output_dict=True, digits=5)
        }

        # Print Best F1 score and parameters
        logging.info(f"Best F1 Score: {f1:.5f} with params: {grid_search.best_params_}")
        logging.info(
            f"Total time: {time.perf_counter() - start_time:.2f} seconds with {ind}th setting: {settings_str}")
        logging.info(f"End time {timestamp}")
        logging.info(f"{'*' * 150}\n\n")

        recall_threshold = 0.87
        print(f"condition: : {recall >= recall_threshold and f1 > best_f1}")

        if recall >= recall_threshold and f1 > best_f1:
            best_f1 = f1
            best_recall = recall
            best_model_name = model_name
            best_settings = settings
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

    visualize_results(results, setting_options)

# Print the best model summary
logging.info(f"\n{'=' * 150}")
logging.info(f"Best Model Summary:")
logging.info(f"Model: {best_model_name}")
logging.info(f"Best F1 Score: {best_f1:.5f}")
logging.info(f"Best Recall: {best_recall:.5f}")
logging.info(f"Best Params: {best_params}")
logging.info(f"Setting: {best_settings}")
logging.info(f"{'=' * 150}")
logging.info(f"Total train + validation time: {time.perf_counter() - overall_start_time:.4f} seconds")
logging.info(f"\n\n\n\n")

strr = f"\tModel Testing Results - {current_time}"
strr_leng = len(strr) + int(15)
logging.info(f"{'*' * strr_leng}")
logging.info(strr)
logging.info(f"{'*' * strr_leng}")
logging.info(f"\n\n\n\n")

# 5. Testing
if best_model:
    # Log evaluation start for the best model
    logging.info(f"\nEvaluating the best model for test: {best_model_name} with setting: {best_settings}")

    # Preprocess test data using the best model's settings
    preprocessor = CustomPreprocessor(
        is_pca=best_settings.get("PCA", False),
        is_SMOTE=best_settings.get("SMOTE", True),
        is_dropping=best_settings.get("Dropped", True),
        is_chi_square_dropping=best_settings.get("ChiSquare", False),
        is_pdays_binary_conveted=best_settings.get("PDAYSBinary", True),
        is_duration_0_removed=best_settings.get("is_duration_0_removed", False),
        is_campaign_previous_duration_capped=best_settings.get("Capped", False),
        is_skewness_handled=best_settings.get("HandleSkewness", True),
        pca_threshold=best_settings.get("pca_threshold", 0.96),
        top_k=best_settings.get("top_k", 16),
        p_value=best_settings.get("p_value", 0.05),
        corr_threshold=best_settings.get("corr_threshold", 0.9)
    )

    # Preprocess the training data
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train.copy(), y_train.copy())
    best_model.fit(X_train_processed, y_train_processed)

    # Preprocess the test data
    X_test_processed = preprocessor.transform(X_test)

    # Evaluate on the test set
    y_test_pred = best_model.predict(X_test_processed.copy())
    test_f1 = f1_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    # Log test set results
    logging.info(f"Test Set Evaluation for Best Model:")
    logging.info(f"Test F1 Score: {test_f1:.5f}, Test Recall: {test_recall:.5f}")
    logging.info(classification_report(y_test, y_test_pred, digits=5))

    # Save the test results
    results[best_model_name][str(best_settings)]["Test F1 Score"] = test_f1
    results[best_model_name][str(best_settings)]["Test Recall"] = test_recall

    # Final visualization after all models are evaluated
    logging.info(f"Total train + validation + test time: {time.perf_counter() - overall_start_time:.4f} seconds")
    