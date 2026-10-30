import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TRAIN_PATH = "mnist_train.csv"
TEST_PATH = "mnist_test.csv"

OUTPUT_DIR = "mnist_outputs_bonus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 10
IMAGE_SIZE = 28
INPUT_SIZE = 784
VALIDATION_RATIO = 0.1

FINAL_MODEL_CONFIG = {
    "hidden_layers": [128, 64],
    "activation": "relu",
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 64,
    "l2_lambda": 0.0001,
    "dropout_keep_probs": [0.90, 0.90],
    "optimizer": "adam",
    "early_stopping": True,
    "patience": 5,
    "min_delta": 1e-4,
}

RUN_HYPERPARAMETER_ANALYSIS = True
RUN_ARCHITECTURE_COMPARISON = True
RUN_ADAM_VS_GD_COMPARISON = True
RUN_DROPOUT_COMPARISON = True

EXPERIMENT_TRAIN_LIMIT = 12000
EXPERIMENT_EPOCHS = 5


# ------------------------------------------------------------
# Data loading and preprocessing
# ------------------------------------------------------------

def read_mnist_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Place {path} next to this script or update TRAIN_PATH / TEST_PATH."
        )

    df = pd.read_csv(path)

    if "label" in df.columns:
        y = df["label"].values.astype(int)
        X = df.drop(columns=["label"]).values.astype(np.float32)
    else:
        if df.shape[1] == 785:
            first_col_numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            valid_label_column = (
                first_col_numeric.isna().sum() == 0
                and set(first_col_numeric.unique()).issubset(set(range(10)))
            )

            if valid_label_column:
                y = df.iloc[:, 0].values.astype(int)
                X = df.iloc[:, 1:].values.astype(np.float32)
            else:
                df = pd.read_csv(path, header=None)
                y = df.iloc[:, 0].values.astype(int)
                X = df.iloc[:, 1:].values.astype(np.float32)
        else:
            df = pd.read_csv(path, header=None)
            y = df.iloc[:, 0].values.astype(int)
            X = df.iloc[:, 1:].values.astype(np.float32)

    X = X.reshape(X.shape[0], -1)

    if X.shape[1] != INPUT_SIZE:
        raise ValueError(f"Expected {INPUT_SIZE} pixel features, got {X.shape[1]} features.")

    X = X / 255.0

    return X, y


def train_validation_split(X, y, validation_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    val_size = int(X.shape[0] * validation_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def one_hot(y, num_classes=10):
    Y = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


# ------------------------------------------------------------
# Activation functions
# ------------------------------------------------------------

def sigmoid(z):
    z = np.asarray(z)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def sigmoid_derivative_from_z(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative_from_z(z):
    return (z > 0).astype(np.float32)


def tanh(z):
    return np.tanh(z)


def tanh_derivative_from_z(z):
    t = np.tanh(z)
    return 1.0 - t ** 2


def get_activation_functions(name):
    name = name.lower()

    if name == "sigmoid":
        return sigmoid, sigmoid_derivative_from_z
    if name == "relu":
        return relu, relu_derivative_from_z
    if name == "tanh":
        return tanh, tanh_derivative_from_z

    raise ValueError("activation must be one of: sigmoid, relu, tanh")


# ------------------------------------------------------------
# Output layer and loss
# ------------------------------------------------------------

def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy_loss(y_true_onehot, y_pred_prob, parameters=None, l2_lambda=0.0):
    eps = 1e-12
    m = y_true_onehot.shape[0]

    data_loss = -np.sum(y_true_onehot * np.log(y_pred_prob + eps)) / m

    reg_loss = 0.0
    if parameters is not None and l2_lambda > 0:
        for key, value in parameters.items():
            if key.startswith("W"):
                reg_loss += np.sum(value ** 2)
        reg_loss *= l2_lambda / (2.0 * m)

    return float(data_loss + reg_loss)


def mse_loss(y_true_onehot, y_pred_prob):
    return float(np.mean((y_true_onehot - y_pred_prob) ** 2))


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix_manual(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        cm[int(actual), int(predicted)] += 1

    return cm


def per_class_accuracy_from_cm(cm):
    result = {}
    for class_id in range(cm.shape[0]):
        support = cm[class_id].sum()
        result[class_id] = cm[class_id, class_id] / support if support > 0 else 0.0
    return result


def advanced_confusion_analysis(cm, top_k=15):
    rows = []

    for true_label in range(cm.shape[0]):
        for pred_label in range(cm.shape[1]):
            if true_label != pred_label and cm[true_label, pred_label] > 0:
                rows.append(
                    {
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(cm[true_label, pred_label]),
                    }
                )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    df = df.sort_values("count", ascending=False).head(top_k).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Optimizers
# ------------------------------------------------------------

class Optimizer:
    def __init__(self, name="gd", learning_rate=0.1, parameters=None):
        self.name = name.lower()
        self.learning_rate = learning_rate
        self.t = 0
        self.m = {}
        self.v = {}

        if parameters is not None:
            for key, value in parameters.items():
                self.m[key] = np.zeros_like(value)
                self.v[key] = np.zeros_like(value)

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def update(self, parameters, gradients):
        if self.name in ["gd", "sgd", "mini_batch_gd"]:
            for layer in range(1, len(parameters) // 2 + 1):
                parameters[f"W{layer}"] -= self.learning_rate * gradients[f"dW{layer}"]
                parameters[f"b{layer}"] -= self.learning_rate * gradients[f"db{layer}"]
            return

        if self.name == "adam":
            self.t += 1

            for layer in range(1, len(parameters) // 2 + 1):
                for param_name, grad_name in [(f"W{layer}", f"dW{layer}"), (f"b{layer}", f"db{layer}")]:
                    grad = gradients[grad_name]

                    if param_name not in self.m:
                        self.m[param_name] = np.zeros_like(parameters[param_name])
                        self.v[param_name] = np.zeros_like(parameters[param_name])

                    self.m[param_name] = self.beta1 * self.m[param_name] + (1.0 - self.beta1) * grad
                    self.v[param_name] = self.beta2 * self.v[param_name] + (1.0 - self.beta2) * (grad ** 2)

                    m_hat = self.m[param_name] / (1.0 - self.beta1 ** self.t)
                    v_hat = self.v[param_name] / (1.0 - self.beta2 ** self.t)

                    parameters[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            return

        raise ValueError("optimizer must be 'gd' or 'adam'")


# ------------------------------------------------------------
# Neural network implementation
# ------------------------------------------------------------

class NeuralNetwork:
    def __init__(
        self,
        layer_sizes,
        activation="relu",
        learning_rate=0.1,
        l2_lambda=0.0,
        dropout_keep_probs=None,
        optimizer="gd",
        seed=42,
    ):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.activation, self.activation_derivative = get_activation_functions(activation)
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_keep_probs = dropout_keep_probs
        self.optimizer_name = optimizer
        self.rng = np.random.default_rng(seed)
        self.parameters = {}
        self.initialize_parameters()
        self.optimizer = Optimizer(
            name=optimizer,
            learning_rate=learning_rate,
            parameters=self.parameters,
        )

        num_hidden_layers = len(layer_sizes) - 2
        if self.dropout_keep_probs is None:
            self.dropout_keep_probs = [1.0] * num_hidden_layers

        if len(self.dropout_keep_probs) != num_hidden_layers:
            raise ValueError(
                f"dropout_keep_probs must have {num_hidden_layers} values, "
                f"got {len(self.dropout_keep_probs)}"
            )

    def initialize_parameters(self):
        self.parameters = {}

        for layer in range(1, len(self.layer_sizes)):
            fan_in = self.layer_sizes[layer - 1]
            fan_out = self.layer_sizes[layer]

            if self.activation_name == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)

            self.parameters[f"W{layer}"] = self.rng.normal(
                loc=0.0,
                scale=scale,
                size=(fan_in, fan_out),
            ).astype(np.float32)
            self.parameters[f"b{layer}"] = np.zeros((1, fan_out), dtype=np.float32)

    def forward(self, X, training=False):
        cache = {"A0": X}
        A = X
        num_layers = len(self.layer_sizes) - 1

        for layer in range(1, num_layers):
            W = self.parameters[f"W{layer}"]
            b = self.parameters[f"b{layer}"]

            Z = A @ W + b
            A = self.activation(Z)

            keep_prob = self.dropout_keep_probs[layer - 1]
            if training and keep_prob < 1.0:
                D = (self.rng.random(A.shape) < keep_prob).astype(np.float32)
                A = (A * D) / keep_prob
                cache[f"D{layer}"] = D
            else:
                cache[f"D{layer}"] = None

            cache[f"Z{layer}"] = Z
            cache[f"A{layer}"] = A

        W_last = self.parameters[f"W{num_layers}"]
        b_last = self.parameters[f"b{num_layers}"]

        Z_last = A @ W_last + b_last
        A_last = softmax(Z_last)

        cache[f"Z{num_layers}"] = Z_last
        cache[f"A{num_layers}"] = A_last

        return A_last, cache

    def backward(self, y_onehot, cache):
        gradients = {}
        m = y_onehot.shape[0]
        num_layers = len(self.layer_sizes) - 1

        A_last = cache[f"A{num_layers}"]
        dZ = (A_last - y_onehot) / m

        for layer in range(num_layers, 0, -1):
            A_prev = cache[f"A{layer - 1}"]
            W = self.parameters[f"W{layer}"]

            dW = A_prev.T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)

            if self.l2_lambda > 0:
                dW += (self.l2_lambda / m) * W

            gradients[f"dW{layer}"] = dW
            gradients[f"db{layer}"] = db

            if layer > 1:
                dA_prev = dZ @ W.T

                keep_prob = self.dropout_keep_probs[layer - 2]
                D = cache.get(f"D{layer - 1}")

                if D is not None and keep_prob < 1.0:
                    dA_prev = (dA_prev * D) / keep_prob

                Z_prev = cache[f"Z{layer - 1}"]
                dZ = dA_prev * self.activation_derivative(Z_prev)

        return gradients

    def update_parameters(self, gradients):
        self.optimizer.update(self.parameters, gradients)

    def predict_proba(self, X, batch_size=2048):
        probabilities = []

        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            probs, _ = self.forward(X[start:end], training=False)
            probabilities.append(probs)

        return np.vstack(probabilities)

    def predict(self, X, batch_size=2048):
        return np.argmax(self.predict_proba(X, batch_size=batch_size), axis=1)

    def save_weights(self, path):
        np.savez(path, **self.parameters)

        metadata = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "learning_rate": self.learning_rate,
            "l2_lambda": self.l2_lambda,
            "dropout_keep_probs": self.dropout_keep_probs,
            "optimizer": self.optimizer_name,
        }

        metadata_path = path.replace(".npz", "_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def load_weights(self, path):
        loaded = np.load(path)
        for key in loaded.files:
            self.parameters[key] = loaded[key]


# ------------------------------------------------------------
# Training loop with Mini-Batch and Early Stopping
# ------------------------------------------------------------

def iterate_minibatches(X, y, batch_size, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_onehot = one_hot(y, NUM_CLASSES)
    y_prob = model.predict_proba(X)

    loss = cross_entropy_loss(
        y_onehot,
        y_prob,
        parameters=model.parameters,
        l2_lambda=model.l2_lambda,
    )
    acc = accuracy_score(y, y_pred)

    return loss, acc, y_pred, y_prob


def copy_parameters(parameters):
    return {key: value.copy() for key, value in parameters.items()}


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    batch_size=128,
    early_stopping=False,
    patience=5,
    min_delta=1e-4,
    monitor="val_loss",
    verbose=True,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_monitor_value = np.inf if monitor == "val_loss" else -np.inf
    best_parameters = copy_parameters(model.parameters)
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        for X_batch, y_batch in iterate_minibatches(
            X_train,
            y_train,
            batch_size=batch_size,
            seed=RANDOM_SEED + epoch,
        ):
            y_batch_onehot = one_hot(y_batch, NUM_CLASSES)
            _, cache = model.forward(X_batch, training=True)
            gradients = model.backward(y_batch_onehot, cache)
            model.update_parameters(gradients)

        train_loss, train_acc, _, _ = evaluate_model(model, X_train, y_train)
        val_loss, val_acc, _, _ = evaluate_model(model, X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        current_monitor_value = val_loss if monitor == "val_loss" else val_acc

        if monitor == "val_loss":
            improved = current_monitor_value < best_monitor_value - min_delta
        else:
            improved = current_monitor_value > best_monitor_value + min_delta

        if improved:
            best_monitor_value = current_monitor_value
            best_parameters = copy_parameters(model.parameters)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

        if early_stopping and patience_counter >= patience:
            if verbose:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {best_epoch}, best {monitor}: {best_monitor_value:.4f}"
                )
            break

    model.parameters = best_parameters

    history["best_epoch"] = best_epoch
    history["early_stopping_used"] = early_stopping

    return history


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_training_history(history, output_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    acc_path = output_path.replace("loss", "accuracy")

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix on Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(NUM_CLASSES))
    plt.yticks(np.arange(NUM_CLASSES))

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_normalized_confusion_matrix(cm, output_path):
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(7, 6))
    plt.imshow(normalized)
    plt.title("Normalized Confusion Matrix on Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(NUM_CLASSES))
    plt.yticks(np.arange(NUM_CLASSES))

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_misclassified_samples(X, y_true, y_pred, output_path, max_samples=12):
    wrong_indices = np.where(y_true != y_pred)[0]

    if len(wrong_indices) == 0:
        print("No misclassified samples found.")
        return

    selected = wrong_indices[:max_samples]
    cols = 4
    rows = int(np.ceil(len(selected) / cols))

    plt.figure(figsize=(cols * 3, rows * 3))

    for k, idx in enumerate(selected):
        plt.subplot(rows, cols, k + 1)
        plt.imshow(X[idx].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
        plt.title(f"True: {y_true[idx]} | Pred: {y_pred[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_experiment_results(df, x_col, y_col, title, output_path):
    plt.figure(figsize=(8, 4.8))

    labels = df[x_col].astype(str).values
    values = df[y_col].values

    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=35, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ------------------------------------------------------------
# Hyperparameter, architecture and bonus experiments
# ------------------------------------------------------------

def make_experiment_subset(X, y, limit):
    if limit is None or limit >= X.shape[0]:
        return X, y

    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = []
    per_class = max(1, limit // NUM_CLASSES)

    for class_id in range(NUM_CLASSES):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)
        selected_indices.extend(class_indices[:per_class])

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]


def run_single_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_layers,
    activation,
    learning_rate,
    batch_size,
    epochs,
    l2_lambda=0.0,
    dropout_keep_probs=None,
    optimizer="gd",
    early_stopping=False,
    patience=3,
    name="experiment",
):
    model = NeuralNetwork(
        layer_sizes=[INPUT_SIZE] + hidden_layers + [NUM_CLASSES],
        activation=activation,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda,
        dropout_keep_probs=dropout_keep_probs,
        optimizer=optimizer,
        seed=RANDOM_SEED,
    )

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        patience=patience,
        verbose=False,
    )

    val_loss, val_acc, _, _ = evaluate_model(model, X_val, y_val)

    return {
        "name": name,
        "hidden_layers": str(hidden_layers),
        "activation": activation,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "l2_lambda": l2_lambda,
        "dropout_keep_probs": str(dropout_keep_probs),
        "optimizer": optimizer,
        "early_stopping": early_stopping,
        "best_epoch": history.get("best_epoch", epochs),
        "validation_loss": val_loss,
        "validation_accuracy": val_acc,
    }


def run_hyperparameter_analysis(X_train, y_train, X_val, y_val):
    X_exp, y_exp = make_experiment_subset(X_train, y_train, EXPERIMENT_TRAIN_LIMIT)

    experiments = []

    learning_rates = [0.01, 0.05, 0.1]
    for lr in learning_rates:
        experiments.append(
            run_single_experiment(
                X_exp,
                y_exp,
                X_val,
                y_val,
                hidden_layers=[64],
                activation="relu",
                learning_rate=lr,
                batch_size=128,
                epochs=EXPERIMENT_EPOCHS,
                optimizer="gd",
                name=f"lr={lr}",
            )
        )

    hidden_neurons = [32, 64, 128]
    for neurons in hidden_neurons:
        experiments.append(
            run_single_experiment(
                X_exp,
                y_exp,
                X_val,
                y_val,
                hidden_layers=[neurons],
                activation="relu",
                learning_rate=0.1,
                batch_size=128,
                epochs=EXPERIMENT_EPOCHS,
                optimizer="gd",
                name=f"hidden={neurons}",
            )
        )

    batch_sizes = [64, 128, 256]
    for batch_size in batch_sizes:
        experiments.append(
            run_single_experiment(
                X_exp,
                y_exp,
                X_val,
                y_val,
                hidden_layers=[64],
                activation="relu",
                learning_rate=0.1,
                batch_size=batch_size,
                epochs=EXPERIMENT_EPOCHS,
                optimizer="gd",
                name=f"batch={batch_size}",
            )
        )

    activations = ["relu", "sigmoid", "tanh"]
    for activation_name in activations:
        experiments.append(
            run_single_experiment(
                X_exp,
                y_exp,
                X_val,
                y_val,
                hidden_layers=[64],
                activation=activation_name,
                learning_rate=0.1,
                batch_size=128,
                epochs=EXPERIMENT_EPOCHS,
                optimizer="gd",
                name=f"activation={activation_name}",
            )
        )

    df = pd.DataFrame(experiments)
    df.to_csv(os.path.join(OUTPUT_DIR, "hyperparameter_analysis.csv"), index=False)

    plot_experiment_results(
        df,
        x_col="name",
        y_col="validation_accuracy",
        title="Hyperparameter Analysis - Validation Accuracy",
        output_path=os.path.join(OUTPUT_DIR, "hyperparameter_analysis_accuracy.png"),
    )

    return df


def run_architecture_comparison(X_train, y_train, X_val, y_val):
    X_exp, y_exp = make_experiment_subset(X_train, y_train, EXPERIMENT_TRAIN_LIMIT)

    configs = [
        {
            "name": "1_hidden_64_relu",
            "hidden_layers": [64],
            "activation": "relu",
            "learning_rate": 0.1,
            "batch_size": 128,
            "optimizer": "gd",
        },
        {
            "name": "2_hidden_128_64_relu",
            "hidden_layers": [128, 64],
            "activation": "relu",
            "learning_rate": 0.1,
            "batch_size": 128,
            "optimizer": "gd",
        },
        {
            "name": "1_hidden_64_sigmoid",
            "hidden_layers": [64],
            "activation": "sigmoid",
            "learning_rate": 0.1,
            "batch_size": 128,
            "optimizer": "gd",
        },
    ]

    rows = []

    for config in configs:
        rows.append(
            run_single_experiment(
                X_exp,
                y_exp,
                X_val,
                y_val,
                hidden_layers=config["hidden_layers"],
                activation=config["activation"],
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                epochs=EXPERIMENT_EPOCHS,
                optimizer=config["optimizer"],
                name=config["name"],
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "architecture_comparison.csv"), index=False)

    plot_experiment_results(
        df,
        x_col="name",
        y_col="validation_accuracy",
        title="Architecture Comparison - Validation Accuracy",
        output_path=os.path.join(OUTPUT_DIR, "architecture_comparison_accuracy.png"),
    )

    return df


def run_dropout_comparison(X_train, y_train, X_val, y_val):
    X_exp, y_exp = make_experiment_subset(X_train, y_train, EXPERIMENT_TRAIN_LIMIT)

    configs = [
        {
            "name": "no_dropout",
            "dropout_keep_probs": [1.0, 1.0],
        },
        {
            "name": "dropout_keep_0.9",
            "dropout_keep_probs": [0.9, 0.9],
        },
        {
            "name": "dropout_keep_0.8",
            "dropout_keep_probs": [0.8, 0.8],
        },
    ]

    rows = []

    for config in configs:
        rows.append(
            run_single_experiment(
                X_exp,
                y_exp,
                X_val,
                y_val,
                hidden_layers=[128, 64],
                activation="relu",
                learning_rate=0.001,
                batch_size=128,
                epochs=EXPERIMENT_EPOCHS,
                dropout_keep_probs=config["dropout_keep_probs"],
                optimizer="adam",
                name=config["name"],
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "dropout_comparison.csv"), index=False)

    plot_experiment_results(
        df,
        x_col="name",
        y_col="validation_accuracy",
        title="Dropout Comparison - Validation Accuracy",
        output_path=os.path.join(OUTPUT_DIR, "dropout_comparison_accuracy.png"),
    )

    return df


def run_adam_vs_gd_comparison(X_train, y_train, X_val, y_val):
    X_exp, y_exp = make_experiment_subset(X_train, y_train, EXPERIMENT_TRAIN_LIMIT)

    configs = [
        {
            "name": "mini_batch_gd_lr_0.1",
            "optimizer": "gd",
            "learning_rate": 0.1,
        },
        {
            "name": "adam_lr_0.001",
            "optimizer": "adam",
            "learning_rate": 0.001,
        },
    ]

    rows = []

    for config in configs:
        rows.append(
            run_single_experiment(
                X_exp,
                y_exp,
                X_val,
                y_val,
                hidden_layers=[128, 64],
                activation="relu",
                learning_rate=config["learning_rate"],
                batch_size=128,
                epochs=EXPERIMENT_EPOCHS,
                optimizer=config["optimizer"],
                name=config["name"],
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "adam_vs_gd_comparison.csv"), index=False)

    plot_experiment_results(
        df,
        x_col="name",
        y_col="validation_accuracy",
        title="Adam vs Mini-Batch Gradient Descent",
        output_path=os.path.join(OUTPUT_DIR, "adam_vs_gd_comparison_accuracy.png"),
    )

    return df


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 70)
    print("MNIST Neural Network Project - Bonus Version")
    print("=" * 70)

    X_train_full, y_train_full = read_mnist_csv(TRAIN_PATH)
    X_test, y_test = read_mnist_csv(TEST_PATH)

    print("Train CSV shape:", X_train_full.shape)
    print("Test CSV shape:", X_test.shape)
    print("Pixel values normalized to [0, 1].")

    X_train, y_train, X_val, y_val = train_validation_split(
        X_train_full,
        y_train_full,
        validation_ratio=VALIDATION_RATIO,
        seed=RANDOM_SEED,
    )

    print("\nSplit sizes:")
    print("Training:", X_train.shape[0])
    print("Validation:", X_val.shape[0])
    print("Test:", X_test.shape[0])

    print("\n" + "=" * 70)
    print("Running hyperparameter analysis")
    print("=" * 70)
    hyper_df = run_hyperparameter_analysis(X_train, y_train, X_val, y_val)
    print(hyper_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("Running architecture comparison")
    print("=" * 70)
    arch_df = run_architecture_comparison(X_train, y_train, X_val, y_val)
    print(arch_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("Running dropout comparison")
    print("=" * 70)
    dropout_df = run_dropout_comparison(X_train, y_train, X_val, y_val)
    print(dropout_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("Running Adam vs Mini-Batch Gradient Descent comparison")
    print("=" * 70)
    adam_df = run_adam_vs_gd_comparison(X_train, y_train, X_val, y_val)
    print(adam_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("Training final model with bonus features")
    print("=" * 70)

    final_layer_sizes = [INPUT_SIZE] + FINAL_MODEL_CONFIG["hidden_layers"] + [NUM_CLASSES]

    model = NeuralNetwork(
        layer_sizes=final_layer_sizes,
        activation=FINAL_MODEL_CONFIG["activation"],
        learning_rate=FINAL_MODEL_CONFIG["learning_rate"],
        l2_lambda=FINAL_MODEL_CONFIG["l2_lambda"],
        dropout_keep_probs=FINAL_MODEL_CONFIG["dropout_keep_probs"],
        optimizer=FINAL_MODEL_CONFIG["optimizer"],
        seed=RANDOM_SEED,
    )

    print("Final architecture:", final_layer_sizes)
    print("Activation:", FINAL_MODEL_CONFIG["activation"])
    print("Learning rate:", FINAL_MODEL_CONFIG["learning_rate"])
    print("Epochs:", FINAL_MODEL_CONFIG["epochs"])
    print("Batch size:", FINAL_MODEL_CONFIG["batch_size"])
    print("L2 lambda:", FINAL_MODEL_CONFIG["l2_lambda"])
    print("Dropout keep probabilities:", FINAL_MODEL_CONFIG["dropout_keep_probs"])
    print("Optimizer:", FINAL_MODEL_CONFIG["optimizer"])
    print("Early stopping:", FINAL_MODEL_CONFIG["early_stopping"])

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=FINAL_MODEL_CONFIG["epochs"],
        batch_size=FINAL_MODEL_CONFIG["batch_size"],
        early_stopping=FINAL_MODEL_CONFIG["early_stopping"],
        patience=FINAL_MODEL_CONFIG["patience"],
        min_delta=FINAL_MODEL_CONFIG["min_delta"],
        monitor="val_loss",
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("Final test evaluation")
    print("=" * 70)

    test_loss, test_acc, y_test_pred, y_test_prob = evaluate_model(model, X_test, y_test)

    print("Test Loss:", f"{test_loss:.4f}")
    print("Test Accuracy:", f"{test_acc:.4f}")

    cm = confusion_matrix_manual(y_test, y_test_pred, NUM_CLASSES)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=np.arange(NUM_CLASSES), columns=np.arange(NUM_CLASSES)))

    per_class_acc = per_class_accuracy_from_cm(cm)
    print("\nPer-class accuracy:")
    for class_id, acc in per_class_acc.items():
        print(f"Digit {class_id}: {acc:.4f}")

    advanced_df = advanced_confusion_analysis(cm, top_k=15)
    print("\nAdvanced Confusion Matrix Analysis - Top Misclassifications:")
    if len(advanced_df) == 0:
        print("No misclassifications found.")
    else:
        print(advanced_df.to_string(index=False))

    if test_acc < 0.80:
        print("\nWARNING: Test accuracy is below 80%. Increase epochs or hidden neurons.")
    else:
        print("\nRequirement satisfied: test accuracy is at least 80%.")

    plot_training_history(
        history,
        output_path=os.path.join(OUTPUT_DIR, "final_model_loss.png"),
    )

    plot_confusion_matrix(
        cm,
        output_path=os.path.join(OUTPUT_DIR, "final_model_confusion_matrix.png"),
    )

    plot_normalized_confusion_matrix(
        cm,
        output_path=os.path.join(OUTPUT_DIR, "final_model_confusion_matrix_normalized.png"),
    )

    plot_misclassified_samples(
        X_test,
        y_test,
        y_test_pred,
        output_path=os.path.join(OUTPUT_DIR, "misclassified_samples.png"),
        max_samples=12,
    )

    model.save_weights(os.path.join(OUTPUT_DIR, "final_model_weights.npz"))

    loaded_model = NeuralNetwork(
        layer_sizes=final_layer_sizes,
        activation=FINAL_MODEL_CONFIG["activation"],
        learning_rate=FINAL_MODEL_CONFIG["learning_rate"],
        l2_lambda=FINAL_MODEL_CONFIG["l2_lambda"],
        dropout_keep_probs=FINAL_MODEL_CONFIG["dropout_keep_probs"],
        optimizer=FINAL_MODEL_CONFIG["optimizer"],
        seed=RANDOM_SEED,
    )
    loaded_model.load_weights(os.path.join(OUTPUT_DIR, "final_model_weights.npz"))
    loaded_test_loss, loaded_test_acc, _, _ = evaluate_model(loaded_model, X_test, y_test)

    print("\nSave/Load check:")
    print("Loaded model test loss:", f"{loaded_test_loss:.4f}")
    print("Loaded model test accuracy:", f"{loaded_test_acc:.4f}")

    history_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_accuracy": history["val_accuracy"],
        }
    )
    history_df.to_csv(os.path.join(OUTPUT_DIR, "final_model_training_history.csv"), index=False)

    pd.DataFrame(cm).to_csv(os.path.join(OUTPUT_DIR, "final_model_confusion_matrix.csv"), index=False)
    advanced_df.to_csv(os.path.join(OUTPUT_DIR, "advanced_confusion_analysis.csv"), index=False)

    final_summary = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "loaded_model_test_loss": loaded_test_loss,
        "loaded_model_test_accuracy": loaded_test_acc,
        "best_epoch": history.get("best_epoch"),
        "final_config": FINAL_MODEL_CONFIG,
    }

    with open(os.path.join(OUTPUT_DIR, "final_model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=4)

    print("\n" + "=" * 70)
    print("Output files saved in:", OUTPUT_DIR)
    print("=" * 70)
    print("1) final_model_loss.png")
    print("2) final_model_accuracy.png")
    print("3) final_model_confusion_matrix.png")
    print("4) final_model_confusion_matrix_normalized.png")
    print("5) misclassified_samples.png")
    print("6) hyperparameter_analysis.csv")
    print("7) hyperparameter_analysis_accuracy.png")
    print("8) architecture_comparison.csv")
    print("9) architecture_comparison_accuracy.png")
    print("10) dropout_comparison.csv")
    print("11) dropout_comparison_accuracy.png")
    print("12) adam_vs_gd_comparison.csv")
    print("13) adam_vs_gd_comparison_accuracy.png")
    print("14) final_model_weights.npz")
    print("15) final_model_weights_metadata.json")
    print("16) final_model_training_history.csv")
    print("17) final_model_confusion_matrix.csv")
    print("18) advanced_confusion_analysis.csv")
    print("19) final_model_summary.json")


if __name__ == "__main__":
    main()
