"""
Data preparation for the two problems: load, clean, split, explore, scale.

Input: the CSV files in ``data/raw/`` written by nn_numpy.dataset
(``banknote_auth.csv`` and ``energy_efficiency.csv``).

Output: NumPy arrays ``X_train, y_train, X_val, y_val, X_test, y_test``
ready for the network, plus (optionally) EDA plots saved through
nn_numpy.plots.Visualizer.

The split is 60 / 20 / 20 (train / validation / test). Feature scaling is
always fit on the training split only and then applied to validation and
test, so no information from those splits leaks into training.
"""

import os

import numpy as np
import pandas as pd

# Import shared project settings:
# - DATASETS_DIR: folder where the CSV files are stored
# - RANDOM_SEED: fixed seed for reproducibility
from nn_numpy.config import DATASETS_DIR, RANDOM_SEED

# Import plotting utilities for EDA
from nn_numpy.plots import Visualizer

# Import our custom standard scaler
from nn_numpy.scalers import StandardScaler


def train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    stratify_col: str | None = None,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly split a dataframe into train / validation / test sets using NumPy only.

    Parameters
    ----------
    df : pandas.DataFrame of shape (n_samples, n_columns)
        Dataframe to split (features and target columns together).
    val_size : float, default=0.2
        Fraction of rows assigned to the validation set.
    test_size : float, default=0.2
        Fraction of rows assigned to the test set. With val_size = 0.2 this
        leaves 60% for training: the project's 60 / 20 / 20 split.
        The remaining rows form the training set.
    stratify_col : str or None, default=None
        If given, the split is done separately inside each value of this
        column, so every split keeps (almost exactly) the same class
        proportions as the full dataset.
    random_seed : int, default=RANDOM_SEED
        Seed for the shuffling, so the split is reproducible.

    Returns
    -------
    df_train : pandas.DataFrame of shape (n_train, n_columns)
        Training rows, in shuffled order, with the original index kept.
    df_val : pandas.DataFrame of shape (n_val, n_columns)
        Validation rows, in shuffled order.
    df_test : pandas.DataFrame of shape (n_test, n_columns)
        Test rows, in shuffled order.

    Notes
    -----
    Processing:
    1. Create a ``numpy.random.RandomState`` from ``random_seed``.
    2. Build the groups of row positions: one group per distinct value of
       ``stratify_col`` when stratifying, or a single group of all rows.
    3. For each group:
       a. shuffle the row positions,
       b. take the first ``round(len(group) * val_size)`` for validation,
       c. take the next ``round(len(group) * test_size)`` for test,
       d. keep the rest for training.
    4. Concatenate each split's positions across groups and shuffle them
       again, so the rows are not ordered by class.
    5. Select the rows with ``df.iloc`` and return the three dataframes.

    Because the sizes are rounded per group, the split sizes can differ by
    a row or so from the exact fractions.
    """
    rng = np.random.RandomState(random_seed)

    # One group per class when stratifying, otherwise a single group of all rows.
    if stratify_col is None:
        groups = [np.arange(len(df))]
    else:
        labels = df[stratify_col].to_numpy()
        groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]

    train_idx, val_idx, test_idx = [], [], []

    # Shuffle each group, then cut it into validation, test and train slices.
    for positions in groups:
        positions = rng.permutation(positions)

        # round() of a Python float already returns an int.
        n_val = round(len(positions) * val_size)
        n_test = round(len(positions) * test_size)

        val_idx.append(positions[:n_val])
        test_idx.append(positions[n_val : n_val + n_test])
        # Train takes the remainder, so the three slices cover every row
        # exactly once whatever round() does at a .5 boundary.
        train_idx.append(positions[n_val + n_test :])

    # Merge the groups and shuffle again, so rows are not ordered by class.
    splits = []
    for idx in (train_idx, val_idx, test_idx):
        merged = rng.permutation(np.concatenate(idx))
        splits.append(df.iloc[merged])

    df_train, df_val, df_test = splits
    return df_train, df_val, df_test


# =========================================================
# 1. CLASSIFICATION PREPROCESSING: Banknote Authentication
# =========================================================
class PreprocessBanknote:
    """
    Handles the full preprocessing pipeline for the Banknote dataset.

    The Banknote Authentication dataset has four numeric features
    (variance, skewness, curtosis, entropy of a wavelet-transformed image)
    and a binary target column ``class`` (0 or 1).

    Responsibilities:
    - load the dataset from disk
    - optionally perform basic sanity checks
    - optionally remove duplicate rows
    - split into train / validation / test sets
    - optionally generate EDA plots
    - optionally scale the feature values using training-set statistics only

    Attributes
    ----------
    banknote_path : str
        Full path of ``data/raw/banknote_auth.csv``.

    Notes
    -----
    Leakage: to avoid even EDA-related leakage concerns, the split is
    performed before any scaling-based visualization. That means:
    - training uses scaler statistics from X_train only
    - EDA scaling comparisons are also based on X_train only
    """

    def __init__(self) -> None:
        """
        Store the path to the Banknote dataset CSV file.

        Parameters
        ----------
        None
            Uses the module-level constant ``DATASETS_DIR``.

        Returns
        -------
        None
            Sets ``self.banknote_path``.

        Notes
        -----
        Processing:
        1. Join ``DATASETS_DIR`` with ``"banknote_auth.csv"`` and store it.
        The file is not read here; that happens in ``load_and_clean`` or
        ``get_data``.
        """
        self.banknote_path = os.path.join(DATASETS_DIR, "banknote_auth.csv")

    def load_and_clean(self) -> pd.DataFrame:
        """
        Load the Banknote dataset, run sanity checks, and remove duplicates.

        Parameters
        ----------
        None
            Reads the CSV at ``self.banknote_path``.

        Returns
        -------
        pandas.DataFrame of shape (n_unique_rows, 5)
            The dataset with duplicate rows removed: four float feature
            columns and the integer ``class`` column.

        Notes
        -----
        Processing:
        1. Read the CSV into a DataFrame.
        2. Drop exact duplicate rows and print the shape before and after
           and the number of rows removed.
        3. Print the number of missing values per column and in total
           (nothing is filled or dropped; this is only a check).
        4. Print ``df.info()`` (column types) and ``df.describe()``
           (summary statistics).
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.banknote_path)

        print("\n--- Banknote: Data Cleaning & Sanity Checks ---")

        # Record the shape before duplicate removal
        shape_before = df.shape
        print(f"Shape BEFORE dropping duplicates: {shape_before}")

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Record the shape after duplicate removal
        shape_after = df.shape
        print(f"Shape AFTER dropping duplicates: {shape_after}")

        # Compute how many rows were removed
        rows_removed = shape_before[0] - shape_after[0]
        print(f"Total duplicate rows removed: {rows_removed}")

        # Print missing-value counts
        print("\n--- Banknote: Null Check ---")
        null_counts = df.isnull().sum()
        print(null_counts)
        print(f"Total missing values: {null_counts.sum()}")

        # Print dataset structure
        print("\n--- Banknote: DataFrame Info ---")
        df.info()

        # Print summary statistics
        print("\n--- Banknote: Descriptive Statistics ---")
        print(df.describe())

        return df

    def split_dataframe(
        self, df: pd.DataFrame, random_seed: int = RANDOM_SEED
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the Banknote dataframe into train / validation / test.

        Parameters
        ----------
        df : pandas.DataFrame of shape (n_samples, 5)
            Banknote dataset (four features and the ``class`` column).
        random_seed : int, default=RANDOM_SEED
            Seed for the shuffling, so the split is reproducible.

        Returns
        -------
        df_train : pandas.DataFrame
            About 60% of the rows.
        df_val : pandas.DataFrame
            About 20% of the rows.
        df_test : pandas.DataFrame
            About 20% of the rows.

        Notes
        -----
        Processing:
        1. Print a section header.
        2. Call ``train_val_test_split`` with ``val_size=0.2``,
           ``test_size=0.2`` and ``stratify_col="class"``.

        Stratification is used because this is a classification problem.
        That helps preserve class balance across the splits.
        """
        print("\n--- Banknote: Splitting Data ---")

        # 60% train, 20% validation, 20% test, stratified by class
        return train_val_test_split(
            df, val_size=0.2, test_size=0.2, stratify_col="class", random_seed=random_seed
        )

    def perform_eda(self, df_train: pd.DataFrame, scale_features: bool = True) -> None:
        """
        Generate EDA plots using the training split only.

        Parameters
        ----------
        df_train : pandas.DataFrame of shape (n_train, 5)
            Training subset of the cleaned Banknote dataset, including the
            ``class`` column.
        scale_features : bool, default=True
            Whether to include the scaling-comparison plot.

        Returns
        -------
        None
            Saves plots to the preprocessing-graphs folder through
            ``Visualizer`` (and shows them if ``Visualizer.SHOW_EDA`` is
            True): the scaling comparison (only when ``scale_features``),
            the pairplot and the class-distribution plot.

        Notes
        -----
        Processing:
        1. Drop the ``class`` column to get the training features.
        2. If ``scale_features`` is True, fit a fresh ``StandardScaler`` on
           those features and plot the distributions before and after
           scaling. This scaler is only used for the plot; ``get_data``
           fits its own.
        3. Plot the classification EDA (pairplot coloured by class and the
           class counts) from the training split.
        """
        print("\n--- Banknote: Generating EDA Plots (Training Split Only) ---")

        # Separate input features from target using only the training subset
        X_train_unscaled = df_train.drop(columns=["class"])

        # Only show scaling comparison when feature scaling is enabled
        if scale_features:
            # Fit scaler on training features only for the scaling comparison plot
            scaler = StandardScaler()
            X_train_scaled_arr = scaler.fit_transform(X_train_unscaled.values)

            # Compare feature distributions before and after scaling.
            # Save this specifically as the classification scaling figure.
            Visualizer.plot_scaling_comparison(
                X_train_unscaled,
                X_train_scaled_arr,
                filename="classification_scaling_comparison.png",
                label="Classification",
            )
        else:
            print("--- Banknote: Skipping scaling comparison plot because scaling is disabled ---")

        # Show classification-specific EDA on training data only.
        # These plots are saved under the preprocessing-graphs folder.
        Visualizer.plot_classification_eda(df_train, target_col="class")

    def get_data(
        self,
        show_eda: bool = False,
        preprocessing_enabled: bool = True,
        scale_features: bool = True,
        random_seed: int = RANDOM_SEED,
        run_eda: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full data-preparation pipeline for the Banknote dataset.

        Parameters
        ----------
        show_eda : bool, default=False
            Whether to display EDA plots on screen (they are saved either
            way when EDA runs). Stored in ``Visualizer.SHOW_EDA``.
        preprocessing_enabled : bool, default=True
            Master switch for preprocessing behavior.
            If False, the dataset is loaded and split, but cleaning, EDA,
            and scaling are skipped.
        scale_features : bool, default=True
            Whether to scale input features using StandardScaler fit on the
            training split only. Ignored when ``preprocessing_enabled`` is
            False.
        random_seed : int, default=RANDOM_SEED
            Seed for the train / validation / test split.
        run_eda : bool, default=True
            Whether to generate (and save) the EDA plots. modeling/train.py turns this
            off for every seed after the first, so a multi-seed run saves one
            set of EDA plots instead of overwriting them once per seed.

        Returns
        -------
        X_train : numpy.ndarray of shape (n_train, 4), dtype float64
            Training features (standardized if scaling is on).
        y_train : numpy.ndarray of shape (n_train, 1), dtype int64
            Training labels, 0 or 1.
        X_val : numpy.ndarray of shape (n_val, 4), dtype float64
            Validation features, scaled with the training statistics.
        y_val : numpy.ndarray of shape (n_val, 1), dtype int64
            Validation labels, 0 or 1.
        X_test : numpy.ndarray of shape (n_test, 4), dtype float64
            Test features, scaled with the training statistics.
        y_test : numpy.ndarray of shape (n_test, 1), dtype int64
            Test labels, 0 or 1.

        Notes
        -----
        Processing:
        1. Load the CSV: with cleaning and sanity checks
           (``load_and_clean``) when preprocessing is enabled, otherwise
           the raw file.
        2. Split 60 / 20 / 20, stratified by ``class``.
        3. Set ``Visualizer.SHOW_EDA`` and, if preprocessing is enabled and
           ``run_eda`` is True, generate the EDA plots from the training
           split.
        4. Separate features from the ``class`` target in each split and
           reshape the targets into column vectors of shape (n, 1).
        5. If preprocessing and scaling are enabled, fit a
           ``StandardScaler`` on ``X_train`` and apply the same mean and
           standard deviation to ``X_val`` and ``X_test``.
        6. Print the final shapes and return the six arrays.
        """
        # ---------------------------------------------------------
        # 1. Load dataset
        # ---------------------------------------------------------
        if preprocessing_enabled:
            # Full behavior: load, clean, sanity-check, remove duplicates
            df = self.load_and_clean()
        else:
            # Minimal behavior: load raw CSV only
            print("\n--- Banknote: Preprocessing disabled ---")
            print("Loading raw dataset without cleaning, EDA, or scaling.")
            df = pd.read_csv(self.banknote_path)

        # ---------------------------------------------------------
        # 2. Split dataset (always required for the training pipeline)
        # ---------------------------------------------------------
        df_train, df_val, df_test = self.split_dataframe(df, random_seed=random_seed)

        # ---------------------------------------------------------
        # 3. EDA display control + EDA generation
        # ---------------------------------------------------------
        Visualizer.SHOW_EDA = show_eda

        if preprocessing_enabled and run_eda:
            self.perform_eda(df_train, scale_features=scale_features)
        elif show_eda:
            print("\n--- Banknote: EDA skipped because preprocessing is disabled ---")

        # ---------------------------------------------------------
        # 4. Separate features and targets
        # ---------------------------------------------------------
        X_train = df_train.drop(columns=["class"]).values
        y_train = df_train["class"].to_numpy().reshape(-1, 1)

        X_val = df_val.drop(columns=["class"]).values
        y_val = df_val["class"].to_numpy().reshape(-1, 1)

        X_test = df_test.drop(columns=["class"]).values
        y_test = df_test["class"].to_numpy().reshape(-1, 1)

        # ---------------------------------------------------------
        # 5. Optional feature scaling
        # ---------------------------------------------------------
        if preprocessing_enabled and scale_features:
            print("\n--- Banknote: Scaling Data ---")

            # Fit scaler only on training features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            # Apply the same training statistics to validation and test
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        else:
            print("\n--- Banknote: Feature scaling skipped ---")

        # ---------------------------------------------------------
        # 6. Final shape summary
        # ---------------------------------------------------------
        print("\n--- Banknote: Final Dataset Split ---")
        print(f"Train set -> X: {X_train.shape}, y: {y_train.shape}")
        print(f"Validation set -> X: {X_val.shape}, y: {y_val.shape}")
        print(f"Test set -> X: {X_test.shape}, y: {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test


# ====================================================
# 2. REGRESSION PREPROCESSING: Energy Efficiency
# ====================================================
class PreprocessEnergy:
    """
    Handles the full preprocessing pipeline for the Energy dataset.

    Responsibilities:
    - load the dataset from disk
    - optionally perform basic sanity checks
    - optionally remove duplicate rows
    - split into train / validation / test sets
    - optionally generate EDA plots
    - optionally scale the feature values using training-set statistics only

    Attributes
    ----------
    energy_path : str
        Full path of ``data/raw/energy_efficiency.csv``.

    Notes
    -----
    The dataset has eight building-shape features and two targets,
    Heating_Load and Cooling_Load. We use Heating_Load as the regression
    target. Cooling_Load is excluded from the input features (it is a
    second target, and using it as an input would leak information).
    """

    def __init__(self) -> None:
        """
        Store the path to the Energy dataset CSV file.

        Parameters
        ----------
        None
            Uses the module-level constant ``DATASETS_DIR``.

        Returns
        -------
        None
            Sets ``self.energy_path``.

        Notes
        -----
        Processing:
        1. Join ``DATASETS_DIR`` with ``"energy_efficiency.csv"`` and store
           it. The file is not read here.
        """
        self.energy_path = os.path.join(DATASETS_DIR, "energy_efficiency.csv")

    def load_and_clean(self) -> pd.DataFrame:
        """
        Load the Energy dataset, run sanity checks, and remove duplicates.

        Parameters
        ----------
        None
            Reads the CSV at ``self.energy_path``.

        Returns
        -------
        pandas.DataFrame of shape (n_unique_rows, 10)
            The dataset with duplicate rows removed: eight feature columns
            plus the Heating_Load and Cooling_Load target columns.

        Notes
        -----
        Processing:
        1. Read the CSV into a DataFrame.
        2. Drop exact duplicate rows and print the shape before and after
           and the number of rows removed.
        3. Print the number of missing values per column and in total
           (nothing is filled or dropped; this is only a check).
        4. Print ``df.info()`` (column types) and ``df.describe()``
           (summary statistics).
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.energy_path)

        print("\n--- Energy: Data Cleaning & Sanity Checks ---")

        # Record the shape before duplicate removal
        shape_before = df.shape
        print(f"Shape BEFORE dropping duplicates: {shape_before}")

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Record the shape after duplicate removal
        shape_after = df.shape
        print(f"Shape AFTER dropping duplicates: {shape_after}")

        # Compute how many rows were removed
        rows_removed = shape_before[0] - shape_after[0]
        print(f"Total duplicate rows removed: {rows_removed}")

        # Print missing-value counts
        print("\n--- Energy: Null Check ---")
        null_counts = df.isnull().sum()
        print(null_counts)
        print(f"Total missing values: {null_counts.sum()}")

        # Print dataset structure
        print("\n--- Energy: DataFrame Info ---")
        df.info()

        # Print summary statistics
        print("\n--- Energy: Descriptive Statistics ---")
        print(df.describe())

        return df

    def split_dataframe(
        self, df: pd.DataFrame, random_seed: int = RANDOM_SEED
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the Energy dataframe into train / validation / test.

        Parameters
        ----------
        df : pandas.DataFrame of shape (n_samples, 10)
            Energy dataset (eight features and both target columns).
        random_seed : int, default=RANDOM_SEED
            Seed for the shuffling, so the split is reproducible.

        Returns
        -------
        df_train : pandas.DataFrame
            About 60% of the rows.
        df_val : pandas.DataFrame
            About 20% of the rows.
        df_test : pandas.DataFrame
            About 20% of the rows.

        Notes
        -----
        Processing:
        1. Print a section header.
        2. Call ``train_val_test_split`` with ``val_size=0.2`` and
           ``test_size=0.2`` and no stratification column.

        This is a regression task with a continuous target, so there are
        no classes to stratify by.
        """
        print("\n--- Energy: Splitting Data ---")

        # 60% train, 20% validation, 20% test (no stratification for regression)
        return train_val_test_split(df, val_size=0.2, test_size=0.2, random_seed=random_seed)

    def perform_eda(self, df_train: pd.DataFrame, scale_features: bool = True) -> None:
        """
        Generate EDA plots using the training split only.

        Parameters
        ----------
        df_train : pandas.DataFrame of shape (n_train, 10)
            Training subset of the cleaned Energy dataset, including both
            target columns.
        scale_features : bool, default=True
            Whether to include the scaling-comparison plot.

        Returns
        -------
        None
            Saves plots to the preprocessing-graphs folder through
            ``Visualizer`` (and shows them if ``Visualizer.SHOW_EDA`` is
            True): the scaling comparison (only when ``scale_features``),
            the Heating_Load distribution and the correlation heatmap.

        Notes
        -----
        Processing:
        1. Drop Heating_Load and Cooling_Load to get the training features.
        2. If ``scale_features`` is True, fit a fresh ``StandardScaler`` on
           those features and plot the distributions before and after
           scaling. This scaler is only used for the plot; ``get_data``
           fits its own.
        3. Plot the regression EDA (target histogram and correlation
           heatmap of all columns) from the training split.

        Why: scaling for the comparison plot is fit only on training
        features, so even the EDA path avoids mixing validation/test
        information into scaling statistics.
        """
        print("\n--- Energy: Generating EDA Plots (Training Split Only) ---")

        # Remove both targets so only input features remain
        X_train_unscaled = df_train.drop(columns=["Heating_Load", "Cooling_Load"])

        # Only show scaling comparison when feature scaling is enabled
        if scale_features:
            # Fit scaler only on training features for the scaling comparison plot
            scaler = StandardScaler()
            X_train_scaled_arr = scaler.fit_transform(X_train_unscaled.values)

            # Show feature distributions before and after scaling.
            # Save this specifically as the regression scaling figure.
            Visualizer.plot_scaling_comparison(
                X_train_unscaled,
                X_train_scaled_arr,
                filename="regression_scaling_comparison.png",
                label="Regression",
            )
        else:
            print("--- Energy: Skipping scaling comparison plot because scaling is disabled ---")

        # Show regression-specific EDA using training data only.
        # These plots are saved under the preprocessing-graphs folder.
        Visualizer.plot_regression_eda(df_train, target_col="Heating_Load")

    def get_data(
        self,
        show_eda: bool = False,
        preprocessing_enabled: bool = True,
        scale_features: bool = True,
        random_seed: int = RANDOM_SEED,
        run_eda: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full data-preparation pipeline for the Energy dataset.

        Parameters
        ----------
        show_eda : bool, default=False
            Whether to display EDA plots on screen (they are saved either
            way when EDA runs). Stored in ``Visualizer.SHOW_EDA``.
        preprocessing_enabled : bool, default=True
            Master switch for preprocessing behavior.
            If False, the dataset is loaded and split, but cleaning, EDA,
            and scaling are skipped.
        scale_features : bool, default=True
            Whether to scale input features using StandardScaler fit on the
            training split only. Ignored when ``preprocessing_enabled`` is
            False.
        random_seed : int, default=RANDOM_SEED
            Seed for the train / validation / test split.
        run_eda : bool, default=True
            Whether to generate (and save) the EDA plots. modeling/train.py turns this
            off for every seed after the first, so a multi-seed run saves one
            set of EDA plots instead of overwriting them once per seed.

        Returns
        -------
        X_train : numpy.ndarray of shape (n_train, 8), dtype float64
            Training features (standardized if scaling is on).
        y_train : numpy.ndarray of shape (n_train, 1), dtype float64
            Training Heating_Load values.
        X_val : numpy.ndarray of shape (n_val, 8), dtype float64
            Validation features, scaled with the training statistics.
        y_val : numpy.ndarray of shape (n_val, 1), dtype float64
            Validation Heating_Load values.
        X_test : numpy.ndarray of shape (n_test, 8), dtype float64
            Test features, scaled with the training statistics.
        y_test : numpy.ndarray of shape (n_test, 1), dtype float64
            Test Heating_Load values.

        Notes
        -----
        Processing:
        1. Load the CSV: with cleaning and sanity checks
           (``load_and_clean``) when preprocessing is enabled, otherwise
           the raw file.
        2. Split 60 / 20 / 20 without stratification.
        3. Set ``Visualizer.SHOW_EDA`` and, if preprocessing is enabled and
           ``run_eda`` is True, generate the EDA plots from the training
           split.
        4. In each split, drop both target columns to get the features and
           take Heating_Load, reshaped to a column vector of shape (n, 1),
           as the target.
        5. If preprocessing and scaling are enabled, fit a
           ``StandardScaler`` on ``X_train`` and apply the same mean and
           standard deviation to ``X_val`` and ``X_test``.
        6. Print the final shapes and return the six arrays.

        The targets are not scaled; the network predicts Heating_Load in
        its original units.
        """
        # ---------------------------------------------------------
        # 1. Load dataset
        # ---------------------------------------------------------
        if preprocessing_enabled:
            # Full behavior: load, clean, sanity-check, remove duplicates
            df = self.load_and_clean()
        else:
            # Minimal behavior: load raw CSV only
            print("\n--- Energy: Preprocessing disabled ---")
            print("Loading raw dataset without cleaning, EDA, or scaling.")
            df = pd.read_csv(self.energy_path)

        # ---------------------------------------------------------
        # 2. Split dataset (always required for the training pipeline)
        # ---------------------------------------------------------
        df_train, df_val, df_test = self.split_dataframe(df, random_seed=random_seed)

        # ---------------------------------------------------------
        # 3. EDA display control + EDA generation
        # ---------------------------------------------------------
        Visualizer.SHOW_EDA = show_eda

        if preprocessing_enabled and run_eda:
            self.perform_eda(df_train, scale_features=scale_features)
        elif show_eda:
            print("\n--- Energy: EDA skipped because preprocessing is disabled ---")

        # ---------------------------------------------------------
        # 4. Separate features and targets
        # ---------------------------------------------------------
        X_train = df_train.drop(columns=["Heating_Load", "Cooling_Load"]).values
        y_train = df_train["Heating_Load"].to_numpy().reshape(-1, 1)

        X_val = df_val.drop(columns=["Heating_Load", "Cooling_Load"]).values
        y_val = df_val["Heating_Load"].to_numpy().reshape(-1, 1)

        X_test = df_test.drop(columns=["Heating_Load", "Cooling_Load"]).values
        y_test = df_test["Heating_Load"].to_numpy().reshape(-1, 1)

        # ---------------------------------------------------------
        # 5. Optional feature scaling
        # ---------------------------------------------------------
        if preprocessing_enabled and scale_features:
            print("\n--- Energy: Scaling Data ---")

            # Fit scaler only on training features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            # Apply the same training statistics to validation and test
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        else:
            print("\n--- Energy: Feature scaling skipped ---")

        # ---------------------------------------------------------
        # 6. Final shape summary
        # ---------------------------------------------------------
        print("\n--- Energy: Final Dataset Split ---")
        print(f"Train set -> X: {X_train.shape}, y: {y_train.shape}")
        print(f"Validation set -> X: {X_val.shape}, y: {y_val.shape}")
        print(f"Test set -> X: {X_test.shape}, y: {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test
