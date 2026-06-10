import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .SpamEmailClassifier import SpamEmailClassifier, default_dataset_path


class SpamClassifierGUI:
    def __init__(self, root: tk.Tk, default_dataset=None):
        self.root = root
        self.root.title("Email Spam Classifier")
        self.root.geometry("860x680")
        self.result = None
        self.dataset_var = tk.StringVar(value=str(default_dataset or default_dataset_path()))
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_percent_var = tk.StringVar(value="0%")
        self.prediction_var = tk.StringVar(
            value="Train or evaluate a model before prediction."
        )

        self._build_layout()

    def _build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        dataset_frame = ttk.LabelFrame(self.root, text="Dataset")
        dataset_frame.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        dataset_frame.columnconfigure(0, weight=1)

        ttk.Entry(dataset_frame, textvariable=self.dataset_var).grid(
            row=0, column=0, sticky="ew", padx=10, pady=10
        )
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).grid(
            row=0, column=1, padx=(0, 10), pady=10
        )
        self.profile_button = ttk.Button(
            dataset_frame, text="Profile Data", command=self.profile_data
        )
        self.profile_button.grid(row=0, column=2, padx=(0, 10), pady=10)
        self.train_button = ttk.Button(
            dataset_frame, text="Train Model", command=self.train_model
        )
        self.train_button.grid(row=0, column=3, padx=(0, 10), pady=10)
        self.evaluate_button = ttk.Button(
            dataset_frame, text="Evaluate Model", command=self.evaluate_model
        )
        self.evaluate_button.grid(row=0, column=4, padx=(0, 10), pady=10)

        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
            length=190,
        )
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=(10, 0))
        self.progress_label = ttk.Label(
            status_frame,
            textvariable=self.progress_percent_var,
            width=5,
            anchor="e",
        )
        self.progress_label.grid(row=0, column=2, sticky="e", padx=(6, 0))
        self.progress_bar.grid_remove()
        self.progress_label.grid_remove()

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))

        metrics_frame = ttk.LabelFrame(body, text="Metrics and Output")
        metrics_frame.rowconfigure(0, weight=1)
        metrics_frame.columnconfigure(0, weight=1)
        self.metrics_text = tk.Text(metrics_frame, wrap="word", height=18)
        self.metrics_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        metrics_scroll = ttk.Scrollbar(
            metrics_frame, orient="vertical", command=self.metrics_text.yview
        )
        metrics_scroll.grid(row=0, column=1, sticky="ns", pady=10)
        self.metrics_text.configure(yscrollcommand=metrics_scroll.set)
        body.add(metrics_frame, weight=3)

        predict_frame = ttk.LabelFrame(body, text="Manual Prediction")
        predict_frame.rowconfigure(1, weight=1)
        predict_frame.columnconfigure(0, weight=1)
        ttk.Label(predict_frame, text="Email Text").grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 4)
        )
        self.message_text = tk.Text(predict_frame, wrap="word", height=8)
        self.message_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        ttk.Button(predict_frame, text="Predict", command=self.predict_message).grid(
            row=2, column=0, sticky="ew", padx=10, pady=(0, 10)
        )
        ttk.Label(
            predict_frame,
            textvariable=self.prediction_var,
            wraplength=280,
            font=("Segoe UI", 11, "bold"),
        ).grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 12))
        body.add(predict_frame, weight=2)

    def browse_dataset(self):
        path = filedialog.askopenfilename(
            title="Select dataset",
            filetypes=[
                ("Supported datasets", "*.csv *.parquet"),
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.dataset_var.set(path)

    def train_model(self):
        self._start_background_job(
            worker=self._train_worker,
            status_text="Training model...",
            intro_text=(
                "Training started. This step trains My Model, compares it with "
                "simple baseline models, and saves metrics, charts, and report files.\n"
            ),
        )

    def profile_data(self):
        try:
            classifier = SpamEmailClassifier(dataset_path=self.dataset_var.get())
            report_text = classifier.profile_data()
        except Exception as exc:
            self._job_failed("Data profiling", str(exc))
            return
        self.status_var.set("Data profile ready")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, report_text)

    def evaluate_model(self):
        self._start_background_job(
            worker=self._evaluate_worker,
            status_text="Evaluating saved model...",
            intro_text=(
                "Evaluation started. This step loads the saved model, computes "
                "test metrics, and refreshes Keras model charts and report files.\n"
            ),
        )

    def _start_background_job(self, worker, status_text: str, intro_text: str):
        self._set_busy_state(True)
        self.status_var.set(status_text)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, intro_text)
        self.progress_bar.grid()
        self.progress_label.grid()
        self._set_progress(0, status_text)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _set_busy_state(self, is_busy: bool):
        state = "disabled" if is_busy else "normal"
        self.train_button.configure(state=state)
        self.evaluate_button.configure(state=state)
        self.profile_button.configure(state=state)
        if not is_busy:
            self.progress_var.set(0)
            self.progress_percent_var.set("0%")
            self.progress_bar.grid_remove()
            self.progress_label.grid_remove()

    def _set_progress(self, percent: float, message: str = ""):
        percent = max(0.0, min(100.0, float(percent)))
        self.progress_var.set(percent)
        self.progress_percent_var.set(f"{percent:.0f}%")
        if message:
            self.status_var.set(message)

    def _progress_callback(self, percent: float, message: str = ""):
        self.root.after(0, lambda: self._set_progress(percent, message))

    def _train_worker(self):
        try:
            result = SpamEmailClassifier(dataset_path=self.dataset_var.get())
            result.train_model(evaluate=False, verbose=False, progress_callback=self._progress_callback)
        except Exception as exc:
            error_message = str(exc)
            self.root.after(0, lambda: self._job_failed("Training", error_message))
            return
        self.root.after(0, lambda: self._job_finished(result))

    def _evaluate_worker(self):
        try:
            result = SpamEmailClassifier(dataset_path=self.dataset_var.get())
            result.evaluate_model(progress_callback=self._progress_callback)
        except Exception as exc:
            error_message = str(exc)
            self.root.after(0, lambda: self._job_failed("Evaluation", error_message))
            return
        self.root.after(0, lambda: self._job_finished(result))

    def _job_finished(self, result):
        self.result = result
        self._set_busy_state(False)
        if result.loaded_from_saved:
            self.status_var.set("Evaluation complete")
        elif result.metrics:
            self.status_var.set("Training and evaluation complete")
        else:
            self.status_var.set("Training complete; evaluate when ready")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, result.report_text)
        self.metrics_text.insert(tk.END, "\n\nGenerated files:\n")
        generated_paths = result.generated_paths or result.output_paths
        for path in generated_paths.values():
            self.metrics_text.insert(tk.END, f"- {path}\n")
        self.prediction_var.set(
            "Evaluation complete. Enter an email and click Predict."
            if result.loaded_from_saved
            else "Model trained. Enter an email to predict, or click Evaluate Model for metrics."
        )

    def _job_failed(self, action_name: str, error_message: str):
        self._set_busy_state(False)
        self.status_var.set(f"{action_name} failed")
        self.metrics_text.insert(tk.END, f"\nError: {error_message}\n")
        messagebox.showerror(f"{action_name} failed", error_message)

    def predict_message(self):
        if self.result is None:
            messagebox.showwarning("Model not ready", "Train or load the model first.")
            return

        message = self.message_text.get("1.0", tk.END).strip()
        if not message:
            messagebox.showwarning("Empty email", "Enter an email to classify.")
            return

        prediction = self.result.predict_message(message)
        self.prediction_var.set(
            f"Prediction: {prediction['label']}\n"
            f"Spam probability: {prediction['probability_spam']:.4f}"
        )


def launch_gui(default_dataset=None):
    root = tk.Tk()
    SpamClassifierGUI(root, default_dataset=default_dataset)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
