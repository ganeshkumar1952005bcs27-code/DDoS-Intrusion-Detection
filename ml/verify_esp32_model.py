import os
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


SCALER_FILE = os.path.join(
    MODEL_DIR, "final_scaler.pkl"
)

PCA_FILE = os.path.join(
    MODEL_DIR, "final_pca_10.pkl"
)

TREE_FILE = os.path.join(
    MODEL_DIR, "final_decision_tree_10pc.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR, "feature_names.json"
)

X_TEST_FILE = os.path.join(
    DATA_DIR, "X_test.csv"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR, "y_test.csv"
)


# ============================================================
# HEADER PARSER
# ============================================================

HEADER_FILE = os.path.join(
    BASE_DIR,
    "esp32",
    "generated_model.h"
)


def parse_float_array(text, name):
    """
    Extract a C float array from generated_model.h.
    """

    start = text.find(name)

    if start == -1:
        raise ValueError(
            f"Could not find array: {name}"
        )

    start = text.find("{", start)
    end = text.find("};", start)

    if start == -1 or end == -1:
        raise ValueError(
            f"Invalid array format: {name}"
        )

    block = text[start + 1:end]

    values = []

    for item in block.replace("\n", " ").split(","):

        item = item.strip()

        if not item:
            continue

        item = item.replace("f", "")

        values.append(float(item))

    return np.array(values, dtype=np.float64)


def parse_int_array(text, name):
    """
    Extract a C integer array from generated_model.h.
    Handles values such as:
        1
        1f
        1.0f
        -1.0f
    """

    start = text.find(name)

    if start == -1:
        raise ValueError(
            f"Could not find array: {name}"
        )

    start = text.find("{", start)

    end = text.find("};", start)

    if start == -1 or end == -1:
        raise ValueError(
            f"Invalid array format: {name}"
        )

    block = text[start + 1:end]

    values = []

    for item in block.replace("\n", " ").split(","):

        item = item.strip()

        if not item:
            continue

        # Remove C/C++ float suffix
        item = item.rstrip("fF")

        # Convert through float first because the current
        # generated header may contain values such as -1.0
        values.append(
            int(float(item))
        )

    return np.array(
        values,
        dtype=np.int32
    )


# ============================================================
# LOAD PYTHON MODELS
# ============================================================

print("=" * 70)
print("STEP 10A - PYTHON vs ESP32 MODEL VERIFICATION")
print("=" * 70)

print("\nLoading Python model...")

scaler = joblib.load(SCALER_FILE)
pca = joblib.load(PCA_FILE)
tree_model = joblib.load(TREE_FILE)

with open(FEATURE_FILE, "r") as f:
    feature_names = json.load(f)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading test data...")

X_test = pd.read_csv(X_TEST_FILE)
y_test = pd.read_csv(
    Y_TEST_FILE
).iloc[:, 0].to_numpy()


X = X_test.to_numpy(dtype=np.float64)


print("\nTest data:")
print("Samples :", X.shape[0])
print("Features:", X.shape[1])


# ============================================================
# LOAD GENERATED HEADER
# ============================================================

print("\nReading generated_model.h...")

with open(
    HEADER_FILE,
    "r",
    encoding="utf-8"
) as f:

    header_text = f.read()


# ============================================================
# EXTRACT EXPORTED PARAMETERS
# ============================================================

print("Extracting exported parameters...")

exported_scaler_mean = parse_float_array(
    header_text,
    "SCALER_MEAN"
)

exported_scaler_scale = parse_float_array(
    header_text,
    "SCALER_SCALE"
)

exported_pca_mean = parse_float_array(
    header_text,
    "PCA_MEAN"
)

print("\nExported parameter sizes:")

print(
    "Scaler mean :",
    len(exported_scaler_mean)
)

print(
    "Scaler scale:",
    len(exported_scaler_scale)
)

print(
    "PCA mean    :",
    len(exported_pca_mean)
)


# ============================================================
# VERIFY SCALER
# ============================================================

print("\nChecking StandardScaler...")

scaler_mean_error = np.max(
    np.abs(
        scaler.mean_
        - exported_scaler_mean
    )
)

scaler_scale_error = np.max(
    np.abs(
        scaler.scale_
        - exported_scaler_scale
    )
)

print(
    "Maximum mean error :",
    scaler_mean_error
)

print(
    "Maximum scale error:",
    scaler_scale_error
)


# ============================================================
# VERIFY PCA MEAN
# ============================================================

print("\nChecking PCA mean...")

pca_mean_error = np.max(
    np.abs(
        pca.mean_
        - exported_pca_mean
    )
)

print(
    "Maximum PCA mean error:",
    pca_mean_error
)


# ============================================================
# PYTHON PREDICTIONS
# ============================================================

print("\nGenerating Python predictions...")

# Use a sample first.
NUM_SAMPLES = min(100, len(X))

X_sample = X[:NUM_SAMPLES]

X_scaled = scaler.transform(
    X_sample
)

X_pca = pca.transform(
    X_scaled
)

python_predictions = tree_model.predict(
    X_pca
).astype(np.int32)


# ============================================================
# IMPLEMENT EXPORTED C PIPELINE IN PYTHON
# ============================================================

print(
    "\nSimulating generated C/C++ pipeline..."
)


# StandardScaler

c_scaled = (
    X_sample
    - exported_scaler_mean
) / exported_scaler_scale


# PCA

exported_pca_components = np.zeros(
    (10, 79),
    dtype=np.float64
)

# Extract PCA_COMPONENTS manually

start = header_text.find(
    "PCA_COMPONENTS"
)

start = header_text.find(
    "{",
    start
)

end = header_text.find(
    "};",
    start
)

pca_block = header_text[
    start + 1:end
]

# Remove braces

pca_block = (
    pca_block
    .replace("{", "")
    .replace("}", "")
)

pca_values = []

for item in pca_block.split(","):

    item = item.strip()

    if not item:
        continue

    item = item.replace("f", "")

    pca_values.append(
        float(item)
    )


exported_pca_components = np.array(
    pca_values,
    dtype=np.float64
).reshape(10, 79)


c_pca = np.zeros(
    (NUM_SAMPLES, 10),
    dtype=np.float64
)

for sample in range(NUM_SAMPLES):

    for component in range(10):

        total = 0.0

        for feature_index in range(79):

            total += (
                (
                    c_scaled[
                        sample,
                        feature_index
                    ]
                    - exported_pca_mean[
                        feature_index
                    ]
                )
                *
                exported_pca_components[
                    component,
                    feature_index
                ]
            )

        c_pca[
            sample,
            component
        ] = total


# ============================================================
# VERIFY PCA
# ============================================================

pca_difference = np.max(
    np.abs(
        X_pca - c_pca
    )
)

print(
    "Maximum PCA output difference:",
    pca_difference
)


# ============================================================
# DECISION TREE USING EXPORTED ARRAYS
# ============================================================

tree_left = parse_int_array(
    header_text,
    "TREE_LEFT"
)

tree_right = parse_int_array(
    header_text,
    "TREE_RIGHT"
)

tree_feature = parse_int_array(
    header_text,
    "TREE_FEATURE"
)

tree_threshold = parse_float_array(
    header_text,
    "TREE_THRESHOLD"
)

tree_class = parse_int_array(
    header_text,
    "TREE_CLASS"
)


def exported_tree_predict(features):

    node = 0

    while (
        tree_left[node] != -1
        and tree_right[node] != -1
    ):

        feature_index = (
            tree_feature[node]
        )

        if (
            features[feature_index]
            <= tree_threshold[node]
        ):
            node = tree_left[node]

        else:
            node = tree_right[node]

    return tree_class[node]


c_predictions = np.array(
    [
        exported_tree_predict(row)
        for row in c_pca
    ],
    dtype=np.int32
)


# ============================================================
# COMPARE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

matching = np.sum(
    python_predictions
    == c_predictions
)

mismatching = (
    NUM_SAMPLES - matching
)

accuracy = (
    matching / NUM_SAMPLES
) * 100


print(
    "\nSamples tested      :",
    NUM_SAMPLES
)

print(
    "Matching predictions:",
    matching
)

print(
    "Different predictions:",
    mismatching
)

print(
    "Prediction agreement :",
    f"{accuracy:.2f}%"
)


# ============================================================
# SHOW DIFFERENCES
# ============================================================

if mismatching > 0:

    print("\nWARNING: Prediction differences found!")

    print("\nFirst differences:")

    count = 0

    for i in range(NUM_SAMPLES):

        if (
            python_predictions[i]
            != c_predictions[i]
        ):

            print(
                f"Sample {i}: "
                f"Python={python_predictions[i]} "
                f"C={c_predictions[i]}"
            )

            count += 1

            if count >= 10:
                break

else:

    print(
        "\nSUCCESS!"
    )

    print(
        "Python and generated C/C++ model "
        "predictions are identical."
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

if (
    mismatching == 0
    and scaler_mean_error < 1e-6
    and scaler_scale_error < 1e-6
    and pca_mean_error < 1e-6
):

    print(
        "STEP 10A COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "STEP 10A REQUIRES INVESTIGATION"
    )

print("=" * 70)