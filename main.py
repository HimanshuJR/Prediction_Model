import sys
import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMRegressor, LGBMClassifier

from utils.helpers import load_data, preprocess_data, create_time_features


class MachineLearningPipeline:
    def __init__(self, model_type='regression', model_path="models/trained_model.pkl"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.encoders = None
        self.model_path = model_path

    def train(self, file_path, target_column, timestamp_column=None, save_model=True, verbose=True):
        if verbose: print("Loading data...")
        try:
            df = load_data(file_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load data: {e}")
        if verbose: print(f"Data loaded: {df.shape}")

        if timestamp_column:
            if verbose: print("Creating time features...")
            df['__original_timestamp__'] = df[timestamp_column]
            df = create_time_features(df, timestamp_column)
            df.drop(columns=[timestamp_column], inplace=True)

        target_cols = target_column if isinstance(target_column, list) else [target_column]
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=target_cols, inplace=True)
        if verbose: print(f"Data after cleaning: {df.shape}")

        if verbose: print("Preprocessing data...")
        data, self.scaler, self.encoders = preprocess_data(df)
        if verbose: print(f"Data after preprocessing: {data.shape}")

        X = data.drop(columns=target_cols)
        y = data[target_cols] if len(target_cols) > 1 else data[target_cols[0]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if verbose: print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        if verbose: print("Training model...")
        if self.model_type == 'multioutput_regression':
            base_model = LGBMRegressor(random_state=42)
            model = MultiOutputRegressor(base_model)
            param_dist = {
                'estimator__n_estimators': [100, 200, 500],
                'estimator__max_depth': [-1, 10, 20],
                'estimator__learning_rate': [0.01, 0.05, 0.1],
                'estimator__num_leaves': [31, 50, 100],
                'estimator__min_child_samples': [20, 50, 100]
            }
            search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=10, cv=3,
                n_jobs=-1, verbose=1, random_state=42
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
        elif self.model_type == 'regression':
            self.model = LGBMRegressor(n_estimators=200, random_state=42)
            self.model.fit(X_train, y_train)
        else:
            self.model = LGBMClassifier(n_estimators=200, random_state=42)
            self.model.fit(X_train, y_train)

        if verbose: print("Evaluating model...")
        self.evaluate(X_test, y_test, target_cols)

        if save_model:
            if verbose: print("Saving model...")
            self.save_model(self.model_path)

        if self.model_type == 'multioutput_regression':
            self.plot_multioutput_feature_importance(X_train)
        else:
            self.plot_feature_importance(X_train)

        if timestamp_column:
            original_timestamps = df.loc[X_test.index, '__original_timestamp__'].reset_index(drop=True)
            X_test_with_ts = X_test.copy()
            X_test_with_ts['timestamp'] = original_timestamps
            X_test_with_ts.to_csv("data/X_test.csv", index=False)
        else:
            X_test.to_csv("data/X_test.csv", index=False)

        if isinstance(y_test, pd.DataFrame):
            y_test.to_csv("data/y_test.csv", index=False)
        else:
            pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

        if verbose: print("Pipeline completed successfully.")

    def evaluate(self, X_test, y_test, target_column=None):
        y_pred = self.model.predict(X_test)

        if self.model_type == 'multioutput_regression':
            print("\nMulti-Output Regression Results:")
            for i, col in enumerate(target_column):
                r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
                mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
                print(f"{col}: R² Score = {r2:.4f}, MSE = {mse:.4f}")
                try:
                    self._plot_actual_vs_predicted(
                        y_test.iloc[:, i].values,
                        y_pred[:, i],
                        f"actual_vs_predicted_{col}.png"
                    )
                except Exception as e:
                    print(f"[ERROR] Plotting failed for {col}: {e}")
        elif self.model_type == 'regression':
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            print(f"\nRegression Results:\nR² Score = {r2:.4f}\nMSE = {mse:.4f}")
            self._plot_actual_vs_predicted(y_test.values, y_pred)
        else:
            acc = accuracy_score(y_test, y_pred)
            print(f"\nClassification Results:\nAccuracy = {acc:.4f}")
            print("Classification Report:\n", classification_report(y_test, y_pred))

    def _plot_actual_vs_predicted(self, y_true, y_pred, filename="actual_vs_predicted.png"):
        plt.figure(figsize=(8, 8))

        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)

        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
            print(f"[WARN] Skipping plot {filename} due to NaN or Inf values.")
            return
        if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
            print(f"[WARN] Skipping plot {filename} due to constant values.")
            return

        plt.scatter(y_true, y_pred, alpha=0.5, label='Data Points')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        try:
            coef = np.polyfit(y_true, y_pred, 1)
            poly1d_fn = np.poly1d(coef)
            plt.plot(y_true, poly1d_fn(y_true), 'b-', label=f'Regression Line (y={coef[0]:.2f}x + {coef[1]:.2f})')
        except Exception as e:
            print(f"[WARN] Skipping regression line for {filename} due to fitting error: {e}")

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_feature_importance(self, X_train, filename="feature_importance.png"):
        if not hasattr(self.model, "feature_importances_"):
            return
        importances = self.model.feature_importances_
        feature_names = X_train.columns
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df.sort_values(by='Importance', ascending=False, inplace=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df.head(10))
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_multioutput_feature_importance(self, X_train, filename="multioutput_feature_importance.png"):
        importances = np.array([est.feature_importances_ for est in self.model.estimators_])
        avg_importance = np.mean(importances, axis=0)

        fi_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': avg_importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df.head(10))
        plt.title("Top 10 Average Feature Importances (Multi-Output)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def predict_and_plot_all(self, file_path, target_columns, timestamp_column=None, output_dir="predictions_all"):
        print("Loading full dataset...")
        df = load_data(file_path)

        if timestamp_column:
            df['__original_timestamp__'] = df[timestamp_column]
            df = create_time_features(df, timestamp_column)
            df.drop(columns=[timestamp_column], inplace=True)

        target_cols = target_columns if isinstance(target_columns, list) else [target_columns]
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=target_cols, inplace=True)

        data, _, _ = preprocess_data(df, fit_scaler=False, scaler=self.scaler, encoders=self.encoders)

        X_full = data.drop(columns=target_cols)
        y_actual = data[target_cols] if len(target_cols) > 1 else data[target_cols[0]]
        y_pred = self.model.predict(X_full)

        os.makedirs(output_dir, exist_ok=True)
        for i, col in enumerate(target_cols):
            try:
                self._plot_actual_vs_predicted(
                    y_actual.iloc[:, i].values,
                    y_pred[:, i],
                    filename=os.path.join(output_dir, f"actual_vs_predicted_full_{col}.png")
                )
            except Exception as e:
                print(f"[ERROR] Could not plot for {col}: {e}")

        pred_df = pd.DataFrame(y_pred, columns=[f"Predicted_{col}" for col in target_cols])
        actual_df = pd.DataFrame(y_actual.values, columns=[f"Actual_{col}" for col in target_cols])
        result_df = pd.concat([actual_df, pred_df], axis=1)

        result_path = os.path.join(output_dir, "actual_vs_predicted_full.csv")
        result_df.to_csv(result_path, index=False)
        print(f"[INFO] Actual vs Predicted results saved to {result_path}")

    def save_model(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'encoders': self.encoders
            }, path)
            print(f"Model saved to {path}")
        except Exception as e:
            raise IOError(f"Failed to save model: {e}")


def main():
    FILE_PATH = "/Users/himanshu/Downloads/machine_data.csv"
    TIMESTAMP_COLUMN = "timestamp"
    SERVO_LOAD_COLUMNS = [
        'ServoLoad_0_path1_JL-10',
        'ServoLoad_1_path1_JL-10',
        'ServoLoad_2_path1_JL-10',
        'ServoLoad_3_path1_JL-10',
        'ServoLoad_4_path1_JL-10',
        'ServoLoad_5_path1_JL-10',
        'ServoLoad_6_path1_JL-10',
        'ServoLoad_7_path1_JL-10',
    ]

    pipeline = MachineLearningPipeline(model_type='multioutput_regression')
    pipeline.train(
        file_path=FILE_PATH,
        target_column=SERVO_LOAD_COLUMNS,
        timestamp_column=TIMESTAMP_COLUMN
    )
    pipeline.predict_and_plot_all(
        file_path=FILE_PATH,
        target_columns=SERVO_LOAD_COLUMNS,
        timestamp_column=TIMESTAMP_COLUMN,
        output_dir="predictions_all"
    )


if __name__ == "__main__":
    main()
