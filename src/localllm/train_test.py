import random
import math
from typing import Any, Optional, Sequence

## Standalone implementation of sklearn's train_test_split - no external dependencies required
def train_test_split(
    *arrays: Sequence[Any],
    test_size: Optional[float | int] = None,
    train_size: Optional[float | int] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[Sequence[Any]] = None,
) -> list:
    """
    Split arrays or sequences into random train and test subsets.

    Replicates the behavior of sklearn.model_selection.train_test_split.

    Parameters
    ----------
    *arrays : sequence of indexables with the same length.
        Allowed inputs are lists, tuples, or any object that supports __len__ and __getitem__.
    test_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
        If int, represents the absolute number of test samples. 
        If None, the value is set to the complement of train_size. Defaults to 0.25 if train_size is also None.
    train_size : float or int, optional
        Complement of test_size. If both are given, they must sum to 1.0 (for floats) or to n_samples (for ints).
    random_state : int, optional
        Seed for the random number generator. Pass an int for reproducible output.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    stratify : sequence, optional
        If not None, data is split in a stratified fashion using this as the class labels (preserves the percentage of samples for each class).

    Returns
    -------
    splitting : list of sequences, length = 2 * len(arrays)
        List containing train-test splits of each input array, in the order:
        X_train, X_test, y_train, y_test, ...

    Examples
    --------
    >>> X = list(range(10))
    >>> y = [i % 2 for i in range(10)]
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    """
    if not arrays:
        raise ValueError("At least one array required as input.")
    n_samples = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != n_samples:
            raise ValueError(
                f"All arrays must have the same length. "
                f"Got lengths: {[len(a) for a in arrays]}"
            )
    n_test, n_train = _validate_sizes(test_size, train_size, n_samples)
    indices = list(range(n_samples))
    if stratify is not None:
        train_idx, test_idx = _stratified_split(indices, stratify, n_train, n_test, random_state)
    else:
        if shuffle:
            rng = random.Random(random_state)
            rng.shuffle(indices)
        train_idx = indices[:n_train]
        test_idx = indices[n_train : n_train + n_test]
    # Apply split to every array
    result = []
    for arr in arrays:
        result.append(_safe_index(arr, train_idx))
        result.append(_safe_index(arr, test_idx))
    return result

def _validate_sizes(
    test_size: Optional[float | int],
    train_size: Optional[float | int],
    n_samples: int,
) -> tuple[int, int]:
    """Return (n_test, n_train) as concrete integer counts."""
    if test_size is None and train_size is None:
        test_size = 0.25
    if test_size is not None and isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size as a float must be in (0, 1).")
        n_test = math.ceil(test_size * n_samples)
    elif test_size is not None:
        n_test = int(test_size)
    else:
        n_test = None  # will be derived from n_train
    if train_size is not None and isinstance(train_size, float):
        if not (0.0 < train_size < 1.0):
            raise ValueError("train_size as a float must be in (0, 1).")
        n_train = math.floor(train_size * n_samples)
    elif train_size is not None:
        n_train = int(train_size)
    else:
        n_train = None  # will be derived from n_test
    if n_test is None:
        n_test = n_samples - n_train
    if n_train is None:
        n_train = n_samples - n_test
    if n_train + n_test > n_samples:
        raise ValueError(f"train_size + test_size = {n_train + n_test} exceeds n_samples = {n_samples}.")
    if n_train <= 0:
        raise ValueError("train_size is too small; results in 0 training samples.")
    if n_test <= 0:
        raise ValueError("test_size is too small; results in 0 test samples.")
    return n_test, n_train


def _safe_index(arr: Sequence[Any], indices: list[int]) -> Any:
    """Index into lists, tuples, or any __getitem__ object."""
    if isinstance(arr, (list, tuple)):
        indexed = [arr[i] for i in indices]
        return type(arr)(indexed) if isinstance(arr, tuple) else indexed
    # numpy-style indexing (for np.ndarray, pd.DataFrame)
    try:
        return arr[indices]
    except (TypeError, KeyError):
        return [arr[i] for i in indices]


def _stratified_split(
    indices: list[int],
    labels: Sequence[Any],
    n_train: int,
    n_test: int,
    random_state: Optional[int],
) -> tuple[list[int], list[int]]:
    """Split indices preserving label proportions."""
    rng = random.Random(random_state)
    # Group indices by class
    class_indices: dict[Any, list[int]] = {}
    for idx in indices:
        label = labels[idx]
        class_indices.setdefault(label, []).append(idx)
    n_samples = len(indices)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label, idx_list in class_indices.items():
        rng.shuffle(idx_list)
        n_class = len(idx_list)
        # Proportional split
        n_class_test = max(1, round(n_class * n_test / n_samples))
        n_class_train = max(1, round(n_class * n_train / n_samples))
        # Clamp so we don't exceed the class size
        n_class_test = min(n_class_test, n_class - 1)
        n_class_train = min(n_class_train, n_class - n_class_test)
        test_idx.extend(idx_list[:n_class_test])
        train_idx.extend(idx_list[n_class_test : n_class_test + n_class_train])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx
