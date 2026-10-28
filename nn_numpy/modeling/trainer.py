"""
Training loop for the NumPy neural network.

Defines `Trainer`, which runs mini-batch gradient descent on a
`NeuralNetwork` with a given loss and optimizer, records the loss history,
optionally stops early on a validation-loss plateau (restoring the best
checkpoint), and evaluates or scores the trained model.

Input: a NeuralNetwork, a loss object, an optimizer, and NumPy arrays for the
training and validation splits.

Output: the trained network (updated in place, best checkpoint restored), the
training history dictionary from fit() (also kept as Trainer.history), and
the metric dictionary from evaluate(). Nothing is written to disk.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np

from nn_numpy.config import RANDOM_SEED
from nn_numpy.nn.layers import BatchNorm
from nn_numpy.nn.losses import BCELoss, MSELoss
from nn_numpy.nn.metrics import binary_cross_entropy, mean_squared_error
from nn_numpy.nn.network import NeuralNetwork
from nn_numpy.nn.optimizers import SGD, AdaBelief, MomentumSGD, Muon


class Trainer:
    """
    Handles neural network training, validation, and test evaluation.

    This class is responsible for:
    - running the epoch loop
    - creating mini-batches
    - computing training and validation loss
    - updating weights using the chosen optimizer
    - tracking metric history
    - optionally applying early stopping
    - evaluating final test performance

    It is designed to work for both:
    - binary classification
    - regression

    Attributes
    ----------
    network : NeuralNetwork
        The model being trained.
    loss_fn : MSELoss or BCELoss
        Loss object used for training and for the reported loss values.
    optimizer : SGD or MomentumSGD or AdaBelief or Muon
        Optimizer with an `update(layer)` method (see nn_numpy.nn.optimizers).
    task_type : str
        "classification" or "regression" (lower-cased).
    random : numpy.random.RandomState
        Generator used to shuffle the training data each epoch.
    history : dict
        Per-epoch results of the most recent `fit()` call.
    """

    def __init__(
        self,
        network: NeuralNetwork,
        loss_fn: MSELoss | BCELoss,
        optimizer: SGD | MomentumSGD | AdaBelief | Muon,
        task_type: str,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        min_epochs_before_early_stop: int = 0,
        random_seed: int = RANDOM_SEED,
        max_grad_norm: float | None = None,
    ) -> None:
        """
        Initialize the trainer and store its settings.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network model to train.
        loss_fn : MSELoss or BCELoss
            Loss function object (from nn_numpy.nn.losses) with:
            - forward(y_true, y_pred) -> float, the scalar loss
            - backward(y_true, y_pred) -> numpy.ndarray, dL/dy_pred
        optimizer : SGD or MomentumSGD or AdaBelief or Muon
            Optimizer object (from nn_numpy.nn.optimizers) with an update(layer)
            method that changes the layer's parameters in place.
        task_type : str
            Either "classification" or "regression" (case-insensitive).
            Selects the metric: BCE for classification, MSE for regression.
        early_stopping : bool, default=False
            Whether to enable early stopping using validation-loss
            patience while also monitoring training loss as an
            overfitting signal.
        patience : int, default=10
            Number of consecutive epochs allowed without a meaningful
            validation-loss improvement. The experiments pass the configs'
            value (also 10); no recorded justification for 10.
        min_delta : float, default=0.0
            Minimum loss decrease required to count as a meaningful
            improvement for both training loss and validation loss. 0 counts
            any decrease; the experiments pass the configs' value.
        min_epochs_before_early_stop : int, default=0
            Minimum number of completed epochs required before early
            stopping is allowed to terminate training. 0 disables the guard;
            the experiments pass the configs' value.
        random_seed : int, default=RANDOM_SEED
            Seed for shuffling the training data each epoch.
        max_grad_norm : float or None, default=None
            If set, the global gradient norm of every mini-batch is clipped
            to this value before the optimizer step (see
            NeuralNetwork.clip_gradients). None trains without clipping.

        Returns
        -------
        None
            Stores the arguments as attributes, creates `self.random` and
            an empty `self.history`.

        Raises
        ------
        ValueError
            If `task_type` (after lower-casing) is not "classification" or
            "regression".

        Notes
        -----
        Processing:
        1. Store the network, loss and optimizer, and lower-case
           `task_type`.
        2. Check that the task type is supported.
        3. Store the early-stopping settings and the gradient-norm cap.
        4. Create a dedicated `numpy.random.RandomState(random_seed)` for
           shuffling, so the batch order is reproducible and independent
           of the network's own generators.
        5. Create an empty history with "train_loss", "val_loss",
           "val_metric" and "grad_norm" lists.
        """
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.task_type = task_type.lower()

        # Make sure task_type is one of the two supported options
        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs_before_early_stop = min_epochs_before_early_stop
        self.max_grad_norm = max_grad_norm

        # Dedicated random generator used for shuffling training data
        # each epoch in a reproducible way
        self.random = np.random.RandomState(random_seed)

        # History dictionary used to store training progress over epochs
        self.history: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": [],
            "grad_norm": [],
        }

    def _shuffle_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Shuffle the dataset at the start of each epoch.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), dtype float64
            Input features.
        y : numpy.ndarray of shape (n_samples, 1)
            Targets (int64 0/1 labels for classification, float64 values
            for regression).

        Returns
        -------
        X_shuffled : numpy.ndarray of shape (n_samples, n_features), dtype float64
            Rows of X in a new random order (a copy; X is not modified).
        y_shuffled : numpy.ndarray of shape (n_samples, 1), same dtype as y
            Rows of y in the same order, so each sample keeps its target.

        Notes
        -----
        Processing:
        1. Draw a random permutation of the indices 0..n_samples-1 from
           `self.random`.
        2. Index X and y with that same permutation.
        """
        indices = self.random.permutation(len(X))
        return X[indices], y[indices]

    def _create_batches(
        self, X: np.ndarray, y: np.ndarray, batch_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Split the dataset into mini-batches.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), dtype float64
            Input features (already shuffled by the caller).
        y : numpy.ndarray of shape (n_samples, 1)
            Targets, aligned row-by-row with X.
        batch_size : int
            Number of samples per mini-batch.

        Yields
        ------
        X_batch : numpy.ndarray of shape (b, n_features), dtype float64
            Consecutive rows of X, where b = batch_size except possibly
            for the last batch.
        y_batch : numpy.ndarray of shape (b, 1), same dtype as y
            The matching rows of y.

        Notes
        -----
        Processing:
        1. Step through the rows in steps of `batch_size`.
        2. Yield the slice [start, start + batch_size) of X and y.

        The last batch is smaller when n_samples is not a multiple of
        batch_size; it is kept, not dropped. The slices are views, not
        copies.
        """
        for start_idx in range(0, len(X), batch_size):
            end_idx = start_idx + batch_size
            yield X[start_idx:end_idx], y[start_idx:end_idx]

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the correct evaluation metric for the current task.

        Parameters
        ----------
        y_true : numpy.ndarray of shape (n_samples, 1)
            True targets (int64 0/1 labels for classification, float64
            values for regression).
        y_pred : numpy.ndarray of shape (n_samples, 1), dtype float64
            Predicted outputs (probabilities for classification).

        Returns
        -------
        float
            Metric value: binary cross-entropy for classification, mean
            squared error for regression.

        Notes
        -----
        Processing:
        1. If `self.task_type` is "classification", return
           `binary_cross_entropy(y_true, y_pred)` from nn_numpy.nn.metrics.
        2. Otherwise return `mean_squared_error(y_true, y_pred)`.
        """
        if self.task_type == "classification":
            return binary_cross_entropy(y_true, y_pred)
        else:
            return mean_squared_error(y_true, y_pred)

    def _get_model_state(self) -> list[dict[str, Any]]:
        """
        Save a copy of the model's current state.

        Parameters
        ----------
        None
            Uses `self.network.get_trainable_layers()`.

        Returns
        -------
        list of dict
            One dictionary per trainable layer, in network order, with:
            - "params": dict of str to numpy.ndarray (float64), copies of
              its trainable parameters (e.g. "weights", "bias", "gamma",
              "beta")
            - "buffers": dict of str to numpy.ndarray (float64), copies of
              its non-trainable state, such as BatchNorm's
              "running_mean" and "running_var" (empty for other layers)

        Notes
        -----
        Processing:
        1. For each trainable layer, read its parameters with
           `get_params()` and, if the layer has `get_buffers()`, its
           buffers.
        2. Copy every array, so later training updates do not change the
           saved values.
        3. Collect one {"params", "buffers"} dict per layer into a list.

        Why: during early stopping, we want to restore the best validation
        checkpoint later. That means we need to save the model whenever
        validation loss meaningfully improves. Buffers must be saved too:
        restored BatchNorm weights paired with later running statistics
        would no longer be the model that achieved the best validation loss.
        """
        state = []
        for layer in self.network.get_trainable_layers():
            buffers = layer.get_buffers() if isinstance(layer, BatchNorm) else {}
            state.append(
                {
                    "params": {name: param.copy() for name, param in layer.get_params().items()},
                    "buffers": {name: buf.copy() for name, buf in buffers.items()},
                }
            )
        return state

    def _set_model_state(self, state: list[dict[str, Any]]) -> None:
        """
        Restore a previously saved model state.

        Parameters
        ----------
        state : list of dict
            Saved state, as returned by _get_model_state(): one dict per
            trainable layer with "params" and "buffers" (each a dict of
            str to numpy.ndarray, dtype float64).

        Returns
        -------
        None
            Overwrites the parameters (and buffers, if saved) of the
            network's trainable layers.

        Notes
        -----
        Processing:
        1. Pair each trainable layer with its saved entry, in order.
        2. Restore its parameters with `layer.set_params(...)`.
        3. If buffers were saved for that layer (BatchNorm), restore them
           with `layer.set_buffers(...)`.

        The state must come from the same architecture, because layers
        and saved entries are matched by position.
        """
        for layer, saved in zip(self.network.get_trainable_layers(), state):
            layer.set_params(saved["params"])
            if isinstance(layer, BatchNorm) and saved["buffers"]:
                layer.set_buffers(saved["buffers"])

    def _is_meaningful_improvement(self, current_loss: float, best_loss: float) -> bool:
        """
        Return True when a loss meaningfully improves beyond min_delta.

        Parameters
        ----------
        current_loss : float
            Newly observed loss value.
        best_loss : float
            Best loss seen so far for the same quantity.

        Returns
        -------
        bool
            True if current_loss is lower than best_loss by more than
            min_delta.

        Notes
        -----
        Processing:
        1. Return `current_loss < best_loss - self.min_delta`.

        With min_delta=0.0 any decrease counts. A larger min_delta ignores
        tiny decreases that are just noise.
        """
        return current_loss < (best_loss - self.min_delta)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Train the network with mini-batch gradient descent.

        Parameters
        ----------
        X_train : numpy.ndarray of shape (n_train, n_features), dtype float64
            Training inputs.
        y_train : numpy.ndarray of shape (n_train, 1)
            Training targets (int64 0/1 labels for classification, float64
            values for regression).
        X_val : numpy.ndarray of shape (n_val, n_features), dtype float64
            Validation inputs, used after every epoch.
        y_val : numpy.ndarray of shape (n_val, 1)
            Validation targets, same dtype convention as y_train.
        epochs : int, default=100
            Maximum number of training epochs. The experiments pass the
            configs' value (also 100); no recorded justification for 100.
        batch_size : int, default=32
            Number of samples per mini-batch. Every experiment passes the
            grid's batch size, so this default is never used in the results;
            no recorded justification for 32.
        verbose : bool, default=True
            Whether to print per-epoch progress (and early-stopping status).

        Returns
        -------
        dict
            Training history (also stored as `self.history`) with keys:
            - "train_loss" (list of float): mean training-batch loss per
              epoch, measured in training mode
            - "val_loss" (list of float): validation loss per epoch,
              measured in evaluation mode
            - "val_metric" (list of float): validation BCE or MSE per epoch
            - "grad_norm" (list of float): the largest global gradient norm
              of any mini-batch in each epoch, measured before clipping
              (inf or NaN once gradients overflow)
            - "epochs_ran" (int): number of epochs actually completed
            - "stopped_early" (bool): True if fewer than `epochs` ran
              (early stopping or divergence)
            - "diverged" (bool): True if training stopped because a loss
              became NaN or infinite
            - "stop_reason" (str): why training ended: "diverged",
              "early_stopping" or "max_epochs"
            - "overfitting_epochs" (int): consecutive epochs, ending at
              the stop, in which the training loss still improved by more
              than `min_delta` while the validation loss did not (the
              overfitting signature); 0 unless early stopping ended the run
            - "best_epoch" (int): 1-based epoch with the best validation
              loss
            - "best_val_loss" (float): that validation loss
            - "min_epochs_before_early_stop" (int): the guard setting used

        Notes
        -----
        Processing:
        1. Reset `self.history` and the early-stopping trackers.
        2. For each epoch:
           a. switch the network to training mode and shuffle the
              training data;
           b. for each mini-batch: forward pass, compute the loss,
              backward pass, measure the global gradient norm (and clip it
              to max_grad_norm if set), then let the optimizer update every
              trainable layer;
           c. average the batch losses into the epoch's training loss;
           d. switch to evaluation mode and compute the validation loss
              and metric on the whole validation set;
           e. append the three values and the epoch's largest gradient
              norm to the history;
           f. stop if either loss is not finite (divergence);
           g. if early stopping is enabled, update the best losses,
              save a checkpoint when validation loss improves, update the
              patience counter, and stop once patience runs out after the
              minimum-epoch guard.
        3. If early stopping saved a checkpoint, restore it.
        4. Record "epochs_ran", "stopped_early", "diverged", "stop_reason",
           "overfitting_epochs" and the guard setting, switch to evaluation
           mode, and record "best_epoch" and "best_val_loss".

        When early stopping is enabled, checkpoint restoration still
        uses the best validation-loss model. Patience is driven by
        validation-loss non-improvement only after the configured
        minimum-epoch guard has been reached, while training loss is
        still monitored to indicate whether that stagnation looks like
        the onset of overfitting.

        Without early stopping, the model from the last epoch is kept, and
        "best_epoch"/"best_val_loss" simply report the minimum of the
        recorded validation losses. The non-finite losses of a diverged
        epoch are kept in the history, and a NaN validation loss there
        makes these two values unreliable (`np.argmin` returns the index
        of the NaN), so check "diverged" before trusting them.
        """
        # Reset history at the start of every fit() call
        # so previous runs do not contaminate the new one
        self.history = {"train_loss": [], "val_loss": [], "val_metric": [], "grad_norm": []}

        # Variables used for early stopping
        best_train_loss = float("inf")
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = None
        epochs_without_val_improvement = 0
        train_improving_without_val_count = 0

        # Set to True if the loss overflows to NaN or infinity.
        diverged = False

        # Main training loop
        for epoch in range(epochs):
            # Training mode: layers such as BatchNorm use batch statistics.
            self.network.train()

            # Shuffle training data at the start of each epoch
            X_train_shuffled, y_train_shuffled = self._shuffle_data(X_train, y_train)

            # Track batch losses so we can average them into one epoch loss,
            # and each batch's gradient norm (before clipping).
            batch_losses = []
            batch_grad_norms = []

            # Mini-batch training
            for X_batch, y_batch in self._create_batches(
                X_train_shuffled, y_train_shuffled, batch_size
            ):
                # Forward pass through the network
                y_pred = self.network.forward(X_batch)

                # Compute loss for this batch
                loss = self.loss_fn.forward(y_batch, y_pred)
                batch_losses.append(loss)

                # Compute gradient of the loss with respect to predictions
                grad_loss = self.loss_fn.backward(y_batch, y_pred)

                # Backpropagate through the whole network
                self.network.backward(grad_loss)

                # Measure the global gradient norm, and clip it if a cap is
                # set, before the optimizer reads the gradients. Measuring
                # alone changes nothing.
                if self.max_grad_norm is not None:
                    batch_grad_norms.append(self.network.clip_gradients(self.max_grad_norm))
                else:
                    batch_grad_norms.append(self.network.gradient_norm())

                # Update all trainable layers using the optimizer
                for layer in self.network.get_trainable_layers():
                    self.optimizer.update(layer)

            # Average batch losses to get one training loss for the epoch.
            # The mean is unweighted, so a smaller last batch counts as much as
            # a full one; that is fine for a monitoring curve, and model
            # selection uses the validation loss, not this value.
            train_loss = float(np.mean(batch_losses))

            # Validation pass after the epoch finishes, in evaluation mode so
            # the result reflects the model as it would be used for prediction.
            self.network.eval()
            y_val_pred = self.network.forward(X_val)
            val_loss = self.loss_fn.forward(y_val, y_val_pred)
            val_metric = self._compute_metric(y_val, y_val_pred)

            # Save epoch results in history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)
            # Largest norm of the epoch; NaN if any batch's norm was NaN.
            self.history["grad_norm"].append(float(np.max(batch_grad_norms)))

            # Stop immediately if the loss overflowed. Once a loss is NaN or
            # infinite, every later update is NaN too, so continuing would
            # only waste epochs until early stopping notices.
            if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
                diverged = True
                if verbose:
                    print(
                        f"\nTraining diverged at epoch {epoch + 1}: the loss is no longer finite."
                    )
                break

            # Optional console output
            if verbose:
                metric_name = "BCE" if self.task_type == "classification" else "MSE"
                print(
                    f"Epoch {epoch + 1:03d}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val {metric_name}: {val_metric:.6f}"
                )

            # Early stopping logic
            if self.early_stopping:
                train_improved = self._is_meaningful_improvement(train_loss, best_train_loss)
                val_improved = self._is_meaningful_improvement(val_loss, best_val_loss)
                current_epoch = epoch + 1
                guard_active = current_epoch <= self.min_epochs_before_early_stop

                # Track the best training loss separately from checkpointing so
                # we can tell whether validation stagnation coincides with the
                # training loss still moving downward.
                if train_improved:
                    best_train_loss = float(train_loss)

                # Checkpoint selection remains validation-based so the restored
                # model is still the one with the best validation loss.
                if val_improved:
                    best_val_loss = float(val_loss)
                    best_epoch = current_epoch
                    best_model_state = self._get_model_state()
                    epochs_without_val_improvement = 0
                    train_improving_without_val_count = 0
                else:
                    if guard_active:
                        # During the warm-up period we still keep tracking the
                        # best checkpoint, but we do not let validation plateaus
                        # accumulate toward early stopping yet.
                        epochs_without_val_improvement = 0
                        train_improving_without_val_count = 0
                    else:
                        # Patience is driven by validation-loss stagnation only
                        # after the minimum-epoch guard has expired.
                        epochs_without_val_improvement += 1

                        # Keep a separate overfitting-style counter for
                        # visibility: training keeps improving while validation
                        # does not.
                        if train_improved:
                            train_improving_without_val_count += 1
                        else:
                            train_improving_without_val_count = 0

                if verbose:
                    guard_status = (
                        f"warming_up_until_epoch_{self.min_epochs_before_early_stop}"
                        if guard_active
                        else "active"
                    )
                    print(
                        f"    [ES] best_train_loss={best_train_loss:.6f} | "
                        f"best_val_loss={best_val_loss:.6f} | "
                        f"train_improved={train_improved} | "
                        f"val_improved={val_improved} | "
                        f"guard={guard_status} | "
                        f"no_val_improve={epochs_without_val_improvement}/{self.patience} | "
                        f"train_without_val={train_improving_without_val_count}"
                    )

                # Stop after patience consecutive epochs without meaningful
                # validation improvement, but only after the minimum-epoch
                # guard has been reached. The separate training counter is kept
                # to interpret whether the run appears to be overfitting.
                if (not guard_active) and epochs_without_val_improvement >= self.patience:
                    if verbose:
                        reason = (
                            "validation loss stopped improving while training loss kept improving"
                            if train_improving_without_val_count > 0
                            else "validation loss stopped improving"
                        )
                        print(
                            f"\nEarly stopping triggered at epoch {current_epoch}: {reason} "
                            f"for {self.patience} consecutive epochs after the "
                            f"minimum {self.min_epochs_before_early_stop}-epoch guard."
                        )
                    break

        # After training ends, restore the best validation checkpoint
        # so final test evaluation uses the best model, not just the last epoch
        if self.early_stopping and best_model_state is not None:
            self._set_model_state(best_model_state)

        # Record how many epochs actually ran
        self.history["epochs_ran"] = len(self.history["train_loss"])
        self.history["min_epochs_before_early_stop"] = self.min_epochs_before_early_stop

        # True if training ended before reaching the requested max epochs
        self.history["stopped_early"] = self.history["epochs_ran"] < epochs

        # True if training stopped because the loss stopped being finite
        self.history["diverged"] = diverged

        # Why the loop ended, in one place so reports don't re-derive it.
        if diverged:
            self.history["stop_reason"] = "diverged"
        elif self.history["stopped_early"]:
            self.history["stop_reason"] = "early_stopping"
        else:
            self.history["stop_reason"] = "max_epochs"

        # The overfitting signature at the stop: consecutive epochs in which
        # training loss still improved while validation loss did not. Only
        # meaningful when early stopping ended the run.
        self.history["overfitting_epochs"] = (
            train_improving_without_val_count
            if self.history["stop_reason"] == "early_stopping"
            else 0
        )

        # Leave the network ready for evaluation and prediction.
        self.network.eval()

        # Save best-epoch information.
        #
        # If early stopping was enabled, use the checkpoint-tracking values
        # directly so the reported best epoch/loss match the restored model.
        #
        # If early stopping was not enabled, simply report the raw minimum
        # validation loss observed in history.
        if self.early_stopping and best_epoch is not None:
            self.history["best_epoch"] = best_epoch
            self.history["best_val_loss"] = best_val_loss
        else:
            self.history["best_epoch"] = int(np.argmin(self.history["val_loss"])) + 1
            self.history["best_val_loss"] = float(min(self.history["val_loss"]))

        return self.history

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True
    ) -> dict[str, float]:
        """
        Evaluate the trained model on the test set and print the result.

        Parameters
        ----------
        X_test : numpy.ndarray of shape (n_test, n_features), dtype float64
            Test inputs.
        y_test : numpy.ndarray of shape (n_test, 1)
            True test targets (int64 0/1 labels for classification, float64
            values for regression).
        verbose : bool, default=True
            Whether to print the result. The experiment grid passes False:
            1,440 runs would add three lines each to the run log.

        Returns
        -------
        dict
            Dictionary containing:
            - "test_loss" (float): the training loss function on the test set
            - "test_metric" (float): the task metric on the test set

        Notes
        -----
        Processing:
        1. Switch the network to evaluation mode and predict the whole
           test set in one forward pass.
        2. Compute the loss with `self.loss_fn` and the metric with
           `_compute_metric`.
        3. Print both values if verbose, and return them.

        The metric depends on the task:
        - classification -> BCE
        - regression -> MSE
        """
        # Forward pass on the test set, in evaluation mode
        self.network.eval()
        y_test_pred = self.network.forward(X_test)

        # Compute test loss and task-specific metric
        test_loss = self.loss_fn.forward(y_test, y_test_pred)
        test_metric = self._compute_metric(y_test, y_test_pred)

        if verbose:
            print("\n--- Test Set Evaluation ---")
            print(f"Test Loss: {test_loss:.6f}")

            if self.task_type == "classification":
                print(f"Test BCE: {test_metric:.6f}")
            else:
                print(f"Test MSE: {test_metric:.6f}")

        return {"test_loss": test_loss, "test_metric": test_metric}

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the task metric on any dataset, in evaluation mode, without printing.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), dtype float64
            Inputs.
        y : numpy.ndarray of shape (n_samples, 1)
            True targets (int64 0/1 labels for classification, float64
            values for regression).

        Returns
        -------
        float
            BCE for classification, MSE for regression.

        Notes
        -----
        Processing:
        1. Predict with `predict(X)`, which switches to evaluation mode.
        2. Compute the task metric with `_compute_metric` and return it
           as a Python float.

        Why: the training loss recorded during fit() is measured in training
        mode, so with dropout it includes the dropout noise and is inflated.
        Scoring the training set here, in evaluation mode, gives a training
        metric that is directly comparable with the test metric, which is
        what an honest overfitting (generalization-gap) comparison needs.
        """
        return float(self._compute_metric(y, self.predict(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference using the trained network.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), dtype float64
            Input data.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 1), dtype float64
            Network predictions: probabilities in (0, 1) for a sigmoid
            output (classification), real values for a linear output
            (regression).

        Notes
        -----
        Processing:
        1. Switch the network to evaluation mode, so BatchNorm uses its
           running statistics and Dropout is disabled.
        2. Run one forward pass on all of X and return the output.

        The network is left in evaluation mode afterwards.
        """
        # Evaluation mode, so predictions do not depend on the batch.
        self.network.eval()
        return self.network.forward(X)
