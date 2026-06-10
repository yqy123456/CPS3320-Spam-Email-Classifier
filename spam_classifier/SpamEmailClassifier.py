import json
import os
import random
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")  # GUI safe backend; charts saved to files instead of popping windows
import matplotlib.pyplot as plt
import seaborn as sns

from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Dense, Input
from keras import regularizers, optimizers, utils as keras_utils


# ---------------------------------------------------------------------------
# Paths used by the GUI
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def default_dataset_path():
    """Default path to the dataset, used by the GUI's file picker."""
    return PROJECT_ROOT / "combined_data.csv"


class SpamEmailClassifier:
    # Hyperparameters (same values as the original NumPy version)
    MAX_FEATURES = 1500       # vocabulary size
    HIDDEN_SIZE = 48          # single hidden layer width
    LEARNING_RATE = 0.001
    EPOCHS = 10
    BATCH_SIZE = 128
    L2 = 0.0001
    TEST_SIZE = 0.2
    SEED = 42

    # 8 hand-crafted spam risk features appended to each input vector
    OFFER_WORDS = {"bonus", "cash", "claim", "congratulations", "free", "gift",
                   "offer", "prize", "reward", "selected", "win", "winner", "won"}
    URGENCY_WORDS = {"before", "expire", "expires", "final", "immediately",
                     "limited", "midnight", "now", "today", "tonight", "urgent"}
    ACTION_WORDS = {"activate", "apply", "call", "click", "confirm", "open",
                    "reply", "send", "text", "update", "verify"}
    MONEY_WORDS = {"account", "bank", "billing", "card", "credit", "deposit",
                   "fee", "loan", "payment", "refund", "rebate", "unpaid", "wallet"}
    DELIVERY_WORDS = {"customs", "delivery", "held", "package", "parcel",
                      "shipping", "sorting"}
    SECURITY_WORDS = {"alert", "code", "login", "locked", "password", "profile",
                      "security", "suspended", "suspension", "verify"}

    def __init__(self, dataset_path=None):
        self._set_random_seeds()
        self.dataset_path = Path(dataset_path) if dataset_path else default_dataset_path()
        self.model = None
        self.vectorizer = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.train_df = None
        self.history = {}   # Dictionary to store model histories
        self.metrics = {}   # Dictionary to store metrics (precision, recall, etc.)
        self.comparison_models = {}

        # ------------------------------------------------------------------
        # GUI-required attributes
        # ------------------------------------------------------------------
        self.report_text = ""
        self.loaded_from_saved = False
        self.engine = "Keras"
        self.generated_paths = {}
        self.output_paths = {
            "class_distribution": OUTPUT_DIR / "class_distribution.png",
            "confusion_matrix": OUTPUT_DIR / "confusion_matrix.png",
            "loss_curve": OUTPUT_DIR / "loss_curve.png",
            "metrics_bar": OUTPUT_DIR / "metrics_bar.png",
            "roc_curve": OUTPUT_DIR / "roc_curve.png",
            "probability_distribution": OUTPUT_DIR / "probability_distribution.png",
            "accuracy_comparison": OUTPUT_DIR / "accuracy_comparison.png",
            "model_comparison": OUTPUT_DIR / "model_comparison.png",
            "test_results": OUTPUT_DIR / "test_results.txt",
            "metrics_json": OUTPUT_DIR / "metrics.json",
            "metrics_table": OUTPUT_DIR / "model_comparison_metrics.csv",
            "training_history": OUTPUT_DIR / "training_history.json",
            "data_processing_report": OUTPUT_DIR / "data_processing_report.txt",
            "model_weights": OUTPUT_DIR / "my_model.keras",
            "vocabulary": OUTPUT_DIR / "vocabulary.json",
        }

    # =======================================================================
    # CORE METHODS (professor style: DiabetesClassifier)
    # =======================================================================
    def read_dataset(self):
        # Read in data using pandas (with error handling for missing file)
        try:
            self.train_df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}\n"
                f"Please make sure 'combined_data.csv' is in the project folder."
            )
        except Exception as exc:
            raise ValueError(f"Could not read dataset: {exc}") from exc

        # Auto-detect text and label columns (spam datasets use many names)
        rename_map = {}
        for col in self.train_df.columns:
            lower = col.lower()
            if lower in ("v2", "message", "email", "sms", "body", "content"):
                rename_map[col] = "text"
            elif lower in ("v1", "category", "class", "target", "spam"):
                rename_map[col] = "label"
        self.train_df = self.train_df.rename(columns=rename_map)

        # Make sure the required columns exist after auto-detection
        if "text" not in self.train_df.columns or "label" not in self.train_df.columns:
            raise ValueError(
                f"Dataset must contain a text column and a label column.\n"
                f"Found columns: {list(self.train_df.columns)}"
            )

        # Keep only the two columns we need
        self.train_df = self.train_df[["text", "label"]].dropna()

        # Convert label from "ham"/"spam" to 0/1 if needed
        if self.train_df["label"].dtype == object:
            self.train_df["label"] = self.train_df["label"].str.lower().map({"ham": 0, "spam": 1})
            self.train_df = self.train_df.dropna(subset=["label"])

        try:
            self.train_df["label"] = self.train_df["label"].astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Label column contains unexpected values that cannot be "
                f"converted to integers (expected ham/spam or 0/1): {exc}"
            ) from exc

        # Final sanity check: must have data left after cleaning
        if len(self.train_df) == 0:
            raise ValueError("Dataset is empty after cleaning. Please check the file content.")
        if self.train_df["label"].nunique() < 2:
            raise ValueError(
                "Dataset only contains one class. Need both ham (0) and spam (1)."
            )

    def separate_data(self):
        # Separate the target column from other columns
        self.train_x = self.train_df["text"]
        self.train_y = self.train_df["label"]

        # Split the data into training and testing sets
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.train_x, self.train_y,
            test_size=self.TEST_SIZE, random_state=self.SEED
        )

    def normalize(self):
        # Convert text into numeric feature vectors using bag-of-words
        # (this is the "normalization" step for text data)
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=self.MAX_FEATURES,
            token_pattern=r"[a-zA-Z]{2,}"
        )
        train_bow = self.vectorizer.fit_transform(self.train_x).toarray()
        test_bow = self.vectorizer.transform(self.test_x).toarray()

        # Append 8 hand-crafted spam risk features to each vector
        # (final input size = MAX_FEATURES + 8 = 1508)
        train_extra = np.array([self._engineered_features(t) for t in self.train_x])
        test_extra = np.array([self._engineered_features(t) for t in self.test_x])
        self.train_x = np.hstack([train_bow, train_extra]).astype(np.float32)
        self.test_x = np.hstack([test_bow, test_extra]).astype(np.float32)

    def build_my_model(self):
        self._set_random_seeds()
        # Build MY model
        # Architecture matches the original version: input -> 48 ReLU -> 1 sigmoid
        self.model = Sequential()
        # First build the input layer, no need to add relu function
        self.model.add(Input(shape=(self.train_x.shape[1],)))
        # Single hidden layer with L2 regularization (same as the NumPy version)
        self.model.add(Dense(
            self.HIDDEN_SIZE,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.L2),
        ))
        # Build the output layer
        self.model.add(Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(self.L2),
        ))

        # Compile the model
        optimizer = optimizers.Adam(learning_rate=self.LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy')
        return self.model

    def train_my_model(self):
        # Train the neural network model
        early_stopping_monitor = EarlyStopping(patience=3)
        history = self.model.fit(
            self.train_x, self.train_y,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stopping_monitor]
        )
        self.history['My Model'] = history  # Store history for the neural network

        # Evaluate the accuracy of the trained model on the test set
        predicted_probabilities = self.model.predict(self.test_x, verbose=0).ravel()
        predicted_labels = (predicted_probabilities >= 0.5).astype(int)
        self.calculate_metrics('My Model', predicted_labels, predicted_probabilities)
        return history

    def train_decision_tree(self):
        # Simple baseline model: one decision tree
        model = DecisionTreeClassifier(random_state=self.SEED)
        return self._train_sklearn_model('Decision Tree', model)

    def train_random_forest(self):
        # Simple stronger baseline: many decision trees voting together
        model = RandomForestClassifier(n_estimators=100, random_state=self.SEED)
        return self._train_sklearn_model('Random Forest', model)

    def train_naive_bayes(self):
        # Simple text baseline that works well with word-count features
        model = MultinomialNB()
        return self._train_sklearn_model('Naive Bayes', model)

    def calculate_metrics(self, model_name, predicted_labels, predicted_probabilities=None):
        # Calculate evaluation metrics and confusion matrix for each model
        accuracy = accuracy_score(self.test_y, predicted_labels)
        precision = precision_score(self.test_y, predicted_labels, zero_division=0)
        recall = recall_score(self.test_y, predicted_labels, zero_division=0)
        f1 = f1_score(self.test_y, predicted_labels, zero_division=0)
        cm = confusion_matrix(self.test_y, predicted_labels)

        # Store metrics in the dictionary
        self.metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        if predicted_probabilities is not None:
            fpr, tpr, _ = roc_curve(self.test_y, predicted_probabilities)
            self.metrics[model_name]['roc_auc'] = auc(fpr, tpr)
            self.metrics[model_name]['predicted_probabilities'] = np.asarray(predicted_probabilities)

    def plot_metrics(self):
        # Save all visualizations that can be derived from the current run.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._plot_class_distribution()
        self._plot_loss_curve()

        if 'My Model' in self.metrics:
            self._plot_confusion_matrix()
            self._plot_metrics_bar()
            self._plot_roc_curve()
            self._plot_probability_distribution()
        if len(self.metrics) > 1:
            self._plot_accuracy_comparison()
            self._plot_model_comparison()

    def train_all_models(self):
        self._set_random_seeds()
        # Read and prepare data once
        print("Reading dataset...")
        self.read_dataset()
        print("Separating data...")
        self.separate_data()
        print("Vectorizing text...")
        self.normalize()

        # Train My Model (the main model required by the project)
        print("Training My Model...")
        self.build_my_model()
        self.train_my_model()

        # Train simple comparison models on the same train/test split
        print("Training Decision Tree...")
        self.train_decision_tree()
        print("Training Random Forest...")
        self.train_random_forest()
        print("Training Naive Bayes...")
        self.train_naive_bayes()

        # Plot and show metrics
        self.plot_metrics()

    # =======================================================================
    # GUI-FACING METHODS (thin wrappers around the core methods above)
    # =======================================================================
    def profile_data(self):
        """Quick dataset summary for the GUI's 'Profile Data' button."""
        try:
            self.read_dataset()
            total = len(self.train_df)
            ham = int((self.train_df["label"] == 0).sum())
            spam = int((self.train_df["label"] == 1).sum())
            lines = [
                "Data Processing Report",
                "=" * 44,
                f"Source file: {self.dataset_path}",
                f"Total rows after cleaning: {total}",
                f"Ham (0): {ham}",
                f"Spam (1): {spam}",
                f"Spam ratio: {spam / max(total, 1):.2%}",
                f"Class distribution chart: {self.output_paths['class_distribution']}",
            ]
            self.report_text = "\n".join(lines)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            self._plot_class_distribution()
            self.output_paths["data_processing_report"].write_text(self.report_text, encoding="utf-8")
            self.generated_paths = {
                "data_processing_report": self.output_paths["data_processing_report"],
                "class_distribution": self.output_paths["class_distribution"],
            }
        except (FileNotFoundError, ValueError):
            # Let user-facing errors (missing dataset, bad columns) propagate as-is
            raise
        except Exception as exc:
            raise RuntimeError(f"Data profiling failed: {exc}") from exc
        return self.report_text

    def train_model(self, evaluate=False, verbose=False, progress_callback=None):
        """GUI 'Train Model' button.

        Runs the full professor-style pipeline (read -> separate -> normalize ->
        build -> train -> plot), then saves the model + vocabulary so
        'Evaluate Model' can reload them later.
        """
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            self._set_random_seeds()
            self.loaded_from_saved = False

            self._notify(progress_callback, 0, "Reading dataset...")
            self.read_dataset()

            self._notify(progress_callback, 15, "Separating data...")
            self.separate_data()

            self._notify(progress_callback, 30, "Vectorizing text...")
            self.normalize()

            self._notify(progress_callback, 45, "Building My Model...")
            self.build_my_model()

            self._notify(progress_callback, 55, "Training My Model...")
            self.train_my_model()

            self._notify(progress_callback, 75, "Training comparison models...")
            self.train_decision_tree()
            self.train_random_forest()
            self.train_naive_bayes()

            self._notify(progress_callback, 90, "Saving model and charts...")
            self.plot_metrics()
            self._save_model_and_vocab()
            self._write_training_history()

            self.report_text = self._build_report("Trained and evaluated this run")
            self.output_paths["test_results"].write_text(self.report_text, encoding="utf-8")
            self._write_metrics_json()
            self._write_metrics_table()
        except (FileNotFoundError, ValueError):
            # Let user-facing errors (missing dataset, bad labels, etc.) propagate as-is
            raise
        except Exception as exc:
            raise RuntimeError(f"Training failed: {exc}") from exc

        self.generated_paths = {
            "test_results": self.output_paths["test_results"],
            "class_distribution": self.output_paths["class_distribution"],
            "confusion_matrix": self.output_paths["confusion_matrix"],
            "loss_curve": self.output_paths["loss_curve"],
            "metrics_bar": self.output_paths["metrics_bar"],
            "roc_curve": self.output_paths["roc_curve"],
            "probability_distribution": self.output_paths["probability_distribution"],
            "accuracy_comparison": self.output_paths["accuracy_comparison"],
            "model_comparison": self.output_paths["model_comparison"],
            "training_history": self.output_paths["training_history"],
            "metrics_json": self.output_paths["metrics_json"],
            "metrics_table": self.output_paths["metrics_table"],
            "model_weights": self.output_paths["model_weights"],
            "vocabulary": self.output_paths["vocabulary"],
        }
        self._notify(progress_callback, 100, "Training complete")
        return self

    def evaluate_model(self, progress_callback=None):
        """GUI 'Evaluate Model' button.

        Loads the saved model and vocabulary, prepares the test set, and
        recomputes metrics + charts without retraining.
        """
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            self.loaded_from_saved = True

            self._notify(progress_callback, 0, "Loading saved model...")
            self._load_model_and_vocab()
            self._load_training_history()

            self._notify(progress_callback, 25, "Reading dataset...")
            self.read_dataset()

            self._notify(progress_callback, 45, "Separating data...")
            self.separate_data()

            self._notify(progress_callback, 60, "Vectorizing text...")
            # Use the loaded vectorizer; do NOT refit (would change the feature space)
            train_bow = self.vectorizer.transform(self.train_x).toarray()
            test_bow = self.vectorizer.transform(self.test_x).toarray()
            train_extra = np.array([self._engineered_features(t) for t in self.train_x])
            test_extra = np.array([self._engineered_features(t) for t in self.test_x])
            self.train_x = np.hstack([train_bow, train_extra]).astype(np.float32)
            self.test_x = np.hstack([test_bow, test_extra]).astype(np.float32)

            self._notify(progress_callback, 80, "Evaluating model...")
            predicted_probabilities = self.model.predict(self.test_x, verbose=0).ravel()
            predicted_labels = (predicted_probabilities >= 0.5).astype(int)
            self.calculate_metrics('My Model', predicted_labels, predicted_probabilities)

            self._notify(progress_callback, 88, "Saving evaluation charts...")
            self.plot_metrics()

            self.report_text = self._build_report("Loaded saved Keras model only")
            self.output_paths["test_results"].write_text(self.report_text, encoding="utf-8")
            self._write_metrics_json()
        except (FileNotFoundError, ValueError):
            # Let user-facing errors (missing model/dataset, bad labels) propagate as-is
            raise
        except Exception as exc:
            raise RuntimeError(f"Evaluation failed: {exc}") from exc

        self.generated_paths = {
            "test_results": self.output_paths["test_results"],
            "class_distribution": self.output_paths["class_distribution"],
            "confusion_matrix": self.output_paths["confusion_matrix"],
            "metrics_bar": self.output_paths["metrics_bar"],
            "roc_curve": self.output_paths["roc_curve"],
            "probability_distribution": self.output_paths["probability_distribution"],
            "metrics_json": self.output_paths["metrics_json"],
        }
        if self.output_paths["loss_curve"].exists():
            self.generated_paths["loss_curve"] = self.output_paths["loss_curve"]
        if self.output_paths["training_history"].exists():
            self.generated_paths["training_history"] = self.output_paths["training_history"]
        self._notify(progress_callback, 100, "Evaluation complete")
        return self

    def predict_message(self, message):
        """GUI 'Predict' button - classify a single email."""
        # Input validation: empty message makes no sense
        if message is None or str(message).strip() == "":
            raise ValueError("Email text is empty. Please enter a message to classify.")

        try:
            if self.model is None or self.vectorizer is None:
                self._load_model_and_vocab()

            bow = self.vectorizer.transform([str(message)]).toarray()
            extra = self._engineered_features(message).reshape(1, -1)
            vector = np.hstack([bow, extra]).astype(np.float32)
            probability = float(self.model.predict(vector, verbose=0)[0, 0])
        except FileNotFoundError:
            # "Please train the model first" message from _load_model_and_vocab
            raise
        except Exception as exc:
            raise RuntimeError(f"Prediction failed: {exc}") from exc

        risk_score = self._risk_score(message)
        return {
            "label": "Spam" if probability >= 0.5 else "Ham",
            "probability_spam": probability,
            "risk_score": risk_score,
        }

    # =======================================================================
    # PRIVATE HELPERS (GUI plumbing only; not part of the professor style)
    # =======================================================================
    def _notify(self, progress_callback, percent, message):
        if progress_callback:
            progress_callback(max(0, min(100, float(percent))), message)

    def _set_random_seeds(self):
        # Keep repeated training runs reproducible for class demos.
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        keras_utils.set_random_seed(self.SEED)

    def _save_model_and_vocab(self):
        self.model.save(self.output_paths["model_weights"])
        vocab = {word: int(index) for word, index in self.vectorizer.vocabulary_.items()}
        self.output_paths["vocabulary"].write_text(
            json.dumps({"vocabulary": vocab}, indent=2), encoding="utf-8"
        )

    def _load_model_and_vocab(self):
        # Load saved model and vocabulary; tell the user clearly if they're missing
        model_path = self.output_paths["model_weights"]
        vocab_path = self.output_paths["vocabulary"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found: {model_path}\n"
                "Direct Evaluate requires outputs/my_model.keras from the project zip. "
                "Extract the whole zip folder first, or click 'Train Model' once to "
                "rebuild the model on this computer."
            )
        try:
            self.model = keras_load_model(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Saved model exists but Keras could not load it: {model_path}\n"
                "This usually means the zip was not extracted completely, the model "
                "file is corrupted, or TensorFlow/Keras versions do not match the "
                "README/requirements.txt environment. Reinstall the requirements, or "
                "click 'Train Model' once to rebuild the model on this computer.\n"
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Saved vocabulary not found: {vocab_path}\n"
                "Direct Evaluate requires outputs/vocabulary.json from the project zip. "
                "Extract the whole zip folder first, or click 'Train Model' once to "
                "rebuild the vocabulary on this computer."
            )
        try:
            payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Saved vocabulary not found: {vocab_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Vocabulary file is corrupted: {vocab_path}\n{exc}") from exc

        if "vocabulary" not in payload:
            raise ValueError(
                f"Vocabulary file has the wrong format: {vocab_path}\n"
                "Please click 'Train Model' once to rebuild it."
            )

        vocab = {str(word): int(index) for word, index in payload["vocabulary"].items()}
        # Rebuild a CountVectorizer with the saved vocabulary so predict_message works
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"[a-zA-Z]{2,}",
            vocabulary=vocab,
        )

    def _train_sklearn_model(self, model_name, model):
        model.fit(self.train_x, self.train_y)
        predicted_labels = model.predict(self.test_x)
        predicted_probabilities = model.predict_proba(self.test_x)[:, 1]
        self.calculate_metrics(model_name, predicted_labels, predicted_probabilities)
        self.comparison_models[model_name] = model
        return model

    def _history_payload(self):
        history = self.history.get('My Model')
        if not history:
            return {}

        raw_history = history.history if hasattr(history, "history") else history
        payload = {}
        for key, values in raw_history.items():
            payload[key] = [float(value) for value in values]
        return payload

    def _plot_loss_curve(self):
        history_payload = self._history_payload()
        train_loss = history_payload.get("loss", [])
        if not train_loss:
            return

        epochs = list(range(1, len(train_loss) + 1))
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_loss, marker="o", linewidth=2, label="Training loss")

        val_loss = history_payload.get("val_loss", [])
        if val_loss:
            val_epochs = list(range(1, len(val_loss) + 1))
            plt.plot(val_epochs, val_loss, marker="s", linewidth=2, label="Validation loss")

        plt.title("Loss Curve for My Model")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.xticks(epochs)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_paths["loss_curve"], dpi=150)
        plt.close()

    def _plot_class_distribution(self):
        """Bar chart of ham vs spam counts in the cleaned dataset."""
        if self.train_df is None or "label" not in self.train_df.columns:
            return

        counts = self.train_df["label"].value_counts().sort_index()
        ham_count = int(counts.get(0, 0))
        spam_count = int(counts.get(1, 0))
        total = ham_count + spam_count
        if total == 0:
            return

        labels = ["Ham (0)", "Spam (1)"]
        values = [ham_count, spam_count]
        colors = ["#4C9AFF", "#FF6B6B"]

        plt.figure(figsize=(7, 5))
        bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
        plt.title("Class Distribution (Ham vs Spam)")
        plt.ylabel("Number of Emails")
        plt.grid(axis="y", alpha=0.25)

        # Annotate bars with count and percentage
        for bar, value in zip(bars, values):
            pct = value / total * 100
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(self.output_paths["class_distribution"], dpi=150)
        plt.close()

    def _plot_confusion_matrix(self):
        """Heat-map of the confusion matrix for My Model."""
        metrics = self.metrics.get("My Model")
        if not metrics:
            return

        cm = metrics["confusion_matrix"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
            annot_kws={"size": 14},
        )
        plt.title("Confusion Matrix - My Model")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(self.output_paths["confusion_matrix"], dpi=150)
        plt.close()

    def _plot_metrics_bar(self):
        """Bar chart comparing Accuracy / Precision / Recall / F1 for My Model."""
        metrics = self.metrics.get("My Model")
        if not metrics:
            return

        names = ["Accuracy", "Precision", "Recall", "F1-score"]
        values = [
            float(metrics["accuracy"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
            float(metrics["f1_score"]),
        ]
        colors = ["#4C9AFF", "#36B37E", "#FFAB00", "#FF6B6B"]

        plt.figure(figsize=(7, 5))
        bars = plt.bar(names, values, color=colors, edgecolor="black", linewidth=0.6)
        plt.title("Evaluation Metrics - My Model")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.05)
        plt.grid(axis="y", alpha=0.25)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(self.output_paths["metrics_bar"], dpi=150)
        plt.close()

    def _plot_roc_curve(self):
        """ROC curve with AUC for My Model."""
        metrics = self.metrics.get("My Model")
        if not metrics:
            return

        probabilities = metrics.get("predicted_probabilities")
        if probabilities is None or self.test_y is None:
            return

        fpr, tpr, _ = roc_curve(self.test_y, probabilities)
        roc_auc = metrics.get("roc_auc", auc(fpr, tpr))

        plt.figure(figsize=(7, 5))
        plt.plot(
            fpr,
            tpr,
            color="#4C9AFF",
            linewidth=2,
            label=f"My Model (AUC = {roc_auc:.4f})",
        )
        plt.title("ROC Curve - My Model")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.25)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.output_paths["roc_curve"], dpi=150)
        plt.close()

    def _plot_probability_distribution(self):
        """Histogram of predicted spam probabilities, split by true class."""
        metrics = self.metrics.get("My Model")
        if not metrics:
            return

        probabilities = metrics.get("predicted_probabilities")
        if probabilities is None or self.test_y is None:
            return

        probabilities = np.asarray(probabilities)
        true_labels = np.asarray(self.test_y)
        ham_probs = probabilities[true_labels == 0]
        spam_probs = probabilities[true_labels == 1]

        plt.figure(figsize=(7, 5))
        bins = np.linspace(0.0, 1.0, 31)
        plt.hist(
            ham_probs,
            bins=bins,
            alpha=0.65,
            color="#4C9AFF",
            edgecolor="black",
            linewidth=0.4,
            label=f"Ham (n={len(ham_probs)})",
        )
        plt.hist(
            spam_probs,
            bins=bins,
            alpha=0.65,
            color="#FF6B6B",
            edgecolor="black",
            linewidth=0.4,
            label=f"Spam (n={len(spam_probs)})",
        )
        plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision threshold = 0.5")
        plt.title("Predicted Spam Probability Distribution")
        plt.xlabel("Predicted P(Spam)")
        plt.ylabel("Count")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_paths["probability_distribution"], dpi=150)
        plt.close()

    def _plot_accuracy_comparison(self):
        """Compare test accuracy across My Model and simple baseline models."""
        if not self.metrics:
            return

        names = list(self.metrics.keys())
        values = [float(self.metrics[name]["accuracy"]) for name in names]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(names, values, color=["#4C9AFF", "#36B37E", "#FFAB00", "#FF6B6B"], edgecolor="black", linewidth=0.6)
        plt.title("Accuracy Comparison Across Models")
        plt.ylabel("Test Accuracy")
        plt.ylim(0.0, 1.05)
        plt.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=15, ha="right")

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(self.output_paths["accuracy_comparison"], dpi=150)
        plt.close()

    def _plot_model_comparison(self):
        """Grouped chart for reliable metrics: Accuracy, Precision, Recall, F1, AUC."""
        if not self.metrics:
            return

        model_names = list(self.metrics.keys())
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        x = np.arange(len(model_names))
        width = 0.15

        plt.figure(figsize=(10, 5.5))
        for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            values = [float(self.metrics[name].get(key, 0.0)) for name in model_names]
            offset = (i - (len(metric_keys) - 1) / 2) * width
            plt.bar(x + offset, values, width, label=label)

        plt.title("Model Comparison on the Same Test Set")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.05)
        plt.xticks(x, model_names, rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        plt.tight_layout()
        plt.savefig(self.output_paths["model_comparison"], dpi=150, bbox_inches="tight")
        plt.close()

    def _write_training_history(self):
        history_payload = self._history_payload()
        if not history_payload:
            return
        self.output_paths["training_history"].write_text(
            json.dumps(history_payload, indent=2), encoding="utf-8"
        )

    def _load_training_history(self):
        try:
            history_payload = json.loads(
                self.output_paths["training_history"].read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return
        except json.JSONDecodeError:
            return
        if isinstance(history_payload, dict):
            self.history['My Model'] = history_payload

    def _build_report(self, run_mode):
        lines = [
            "Email Spam Classifier - Test Results",
            "=" * 44,
            f"Dataset: {self.dataset_path}",
            f"Run mode: {run_mode}",
            "",
            "Main Model: Keras Sequential Neural Network",
            f"Architecture: {self.MAX_FEATURES} + 8 -> {self.HIDDEN_SIZE} (ReLU) -> 1 (sigmoid)",
            f"Optimizer: Adam (lr={self.LEARNING_RATE}) | Loss: binary_crossentropy | L2: {self.L2}",
            f"Epochs: {self.EPOCHS} | Batch size: {self.BATCH_SIZE}",
            "",
        ]
        history_payload = self._history_payload()
        train_loss = history_payload.get("loss", [])
        if train_loss:
            lines += [
                "Training Loss:",
                f"Final train loss: {train_loss[-1]:.4f}",
            ]
            val_loss = history_payload.get("val_loss", [])
            if val_loss:
                lines.append(f"Final validation loss: {val_loss[-1]:.4f}")
            lines += [
                f"Loss curve: {self.output_paths['loss_curve']}",
                "",
            ]
        for name, m in self.metrics.items():
            cm = m['confusion_matrix']
            # cm layout: [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            lines += [
                f"--- {name} ---",
                f"Accuracy:  {m['accuracy']:.4f}",
                f"Precision: {m['precision']:.4f}",
                f"Recall:    {m['recall']:.4f}",
                f"F1-score:  {m['f1_score']:.4f}",
                f"ROC-AUC:   {m.get('roc_auc', 0.0):.4f}",
                f"TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}",
                "",
            ]
        if len(self.metrics) > 1:
            lines += [
                f"Accuracy comparison chart: {self.output_paths['accuracy_comparison']}",
                f"Full metrics comparison chart: {self.output_paths['model_comparison']}",
                f"Metrics table: {self.output_paths['metrics_table']}",
                "",
            ]
        return "\n".join(lines)

    def _write_metrics_json(self):
        payload = {}
        for name, m in self.metrics.items():
            payload[name] = {
                "accuracy": float(m["accuracy"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1_score": float(m["f1_score"]),
                "roc_auc": float(m.get("roc_auc", 0.0)),
                "confusion_matrix": m["confusion_matrix"].tolist(),
            }
        self.output_paths["metrics_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_metrics_table(self):
        rows = []
        for name, m in self.metrics.items():
            rows.append({
                "model": name,
                "accuracy": float(m["accuracy"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1_score": float(m["f1_score"]),
                "roc_auc": float(m.get("roc_auc", 0.0)),
            })
        pd.DataFrame(rows).to_csv(self.output_paths["metrics_table"], index=False)

    def _risk_score(self, message):
        """Simple keyword-based risk score for display in the GUI."""
        risky_words = {
            "free", "win", "winner", "won", "prize", "cash", "claim", "urgent",
            "click", "verify", "account", "password", "offer", "congratulations",
            "limited", "expire", "now", "today",
        }
        tokens = set(str(message).lower().split())
        return float(len(tokens & risky_words))

    def _engineered_features(self, text):
        """Build the 8 hand-crafted spam risk features used in the original
        version. Appended to the bag-of-words vector to enrich the input.
        """
        tokens = set(str(text).lower().split())
        has_url = float(any("http" in t or "www." in t for t in tokens))
        has_digit = float(any(any(c.isdigit() for c in t) for t in tokens))
        return np.array([
            has_url,
            has_digit,
            float(bool(tokens & self.OFFER_WORDS)),
            float(bool(tokens & self.URGENCY_WORDS)),
            float(bool(tokens & self.ACTION_WORDS)),
            float(bool(tokens & self.MONEY_WORDS)),
            float(bool(tokens & self.DELIVERY_WORDS)),
            float(bool(tokens & self.SECURITY_WORDS)),
        ], dtype=np.float32)
