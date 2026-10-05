import os
import json
import joblib
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "esp32"
)


SCALER_FILE = os.path.join(
    MODEL_DIR,
    "final_scaler.pkl"
)

PCA_FILE = os.path.join(
    MODEL_DIR,
    "final_pca_10.pkl"
)

TREE_FILE = os.path.join(
    MODEL_DIR,
    "final_decision_tree_10pc.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "generated_model.h"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 10 - ESP32 MODEL EXPORT")
print("=" * 70)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading final model files...")

scaler = joblib.load(
    SCALER_FILE
)

pca = joblib.load(
    PCA_FILE
)

tree_model = joblib.load(
    TREE_FILE
)

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_names = json.load(f)


# ============================================================
# VERIFY MODEL DIMENSIONS
# ============================================================

print("\nModel verification:")

print(
    "Number of input features :",
    len(feature_names)
)

print(
    "Scaler features          :",
    scaler.n_features_in_
)

print(
    "PCA input features       :",
    pca.n_features_in_
)

print(
    "PCA components           :",
    pca.n_components_
)

print(
    "Tree depth               :",
    tree_model.get_depth()
)

print(
    "Tree nodes               :",
    tree_model.tree_.node_count
)


# ============================================================
# SAFETY CHECKS
# ============================================================

if len(feature_names) != 79:

    raise ValueError(
        f"Expected 79 features, "
        f"but found {len(feature_names)}"
    )


if scaler.n_features_in_ != 79:

    raise ValueError(
        "Scaler does not contain 79 input features."
    )


if pca.n_features_in_ != 79:

    raise ValueError(
        "PCA does not expect 79 input features."
    )


if pca.n_components_ != 10:

    raise ValueError(
        "Final PCA model is not a 10-component PCA."
    )


# ============================================================
# HELPER FUNCTION - FLOAT ARRAY
# ============================================================

def write_float_array(
    f,
    c_type,
    name,
    values,
    values_per_line=6
):
    """
    Write a floating-point C/C++ array.

    Uses high precision to minimize differences between
    Python and ESP32/C++ inference.
    """

    values = np.asarray(
        values
    ).flatten()

    f.write(
        f"const {c_type} {name}"
        f"[{len(values)}] = {{\n"
    )

    for i in range(
        0,
        len(values),
        values_per_line
    ):

        chunk = values[
            i:i + values_per_line
        ]

        formatted_values = []

        for value in chunk:

            value = float(value)

            if not np.isfinite(value):

                value = 0.0

            formatted_values.append(
                f"{value:.17g}f"
            )

        line = "    "

        line += ", ".join(
            formatted_values
        )

        if (
            i + values_per_line
            < len(values)
        ):

            line += ","

        f.write(
            line + "\n"
        )

    f.write(
        "};\n\n"
    )


# ============================================================
# HELPER FUNCTION - INTEGER ARRAY
# ============================================================

def write_int_array(
    f,
    c_type,
    name,
    values,
    values_per_line=12
):
    """
    Write an integer C/C++ array.

    Tree indexes and class labels must NOT have
    floating-point suffixes such as '1f'.
    """

    values = np.asarray(
        values
    ).flatten()

    f.write(
        f"const {c_type} {name}"
        f"[{len(values)}] = {{\n"
    )

    for i in range(
        0,
        len(values),
        values_per_line
    ):

        chunk = values[
            i:i + values_per_line
        ]

        formatted_values = []

        for value in chunk:

            formatted_values.append(
                str(int(value))
            )

        line = "    "

        line += ", ".join(
            formatted_values
        )

        if (
            i + values_per_line
            < len(values)
        ):

            line += ","

        f.write(
            line + "\n"
        )

    f.write(
        "};\n\n"
    )


# ============================================================
# EXTRACT DECISION TREE
# ============================================================

tree = tree_model.tree_

children_left = tree.children_left
children_right = tree.children_right
tree_features = tree.feature
threshold = tree.threshold
value = tree.value

node_count = tree.node_count


# ============================================================
# EXTRACT TREE PREDICTED CLASSES
# ============================================================

tree_classes = []

for i in range(
    node_count
):

    class_counts = value[i][0]

    predicted_class = int(
        np.argmax(
            class_counts
        )
    )

    tree_classes.append(
        predicted_class
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# GENERATE HEADER
# ============================================================

print(
    "\nGenerating C/C++ header..."
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    f.write(
r"""#ifndef GENERATED_MODEL_H
#define GENERATED_MODEL_H

/*
 * ============================================================
 * ESP32 TinyML Intrusion Detection Model
 * ============================================================
 *
 * Dataset:
 * IoTID20
 *
 * Input:
 * 79 network-flow features
 *
 * Preprocessing:
 * StandardScaler
 *
 * Dimensionality Reduction:
 * PCA - 10 components
 *
 * Classifier:
 * Decision Tree
 *
 * Classes:
 * 0 = Normal
 * 1 = Attack
 *
 * ============================================================
 * IMPORTANT
 * ============================================================
 *
 * This file is automatically generated from the final
 * Python TinyML model.
 *
 * Do not manually modify the model parameters.
 *
 * ============================================================
 */

#include <math.h>
#include <stdint.h>


"""
    )

    # --------------------------------------------------------
    # MODEL DIMENSIONS
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// MODEL DIMENSIONS\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    f.write(
        "#define NUM_FEATURES 79\n"
    )

    f.write(
        "#define NUM_COMPONENTS 10\n"
    )

    f.write(
        f"#define NUM_TREE_NODES {node_count}\n"
    )

    f.write(
        "\n"
    )


    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// ORIGINAL INPUT FEATURE NAMES\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    f.write(
        "const char* FEATURE_NAMES[NUM_FEATURES] = {\n"
    )

    for i, name in enumerate(
        feature_names
    ):

        comma = ","

        if i == len(feature_names) - 1:

            comma = ""

        f.write(
            f'    "{name}"{comma}\n'
        )

    f.write(
        "};\n\n"
    )


    # --------------------------------------------------------
    # STANDARD SCALER MEAN
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// STANDARD SCALER - MEAN\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    write_float_array(
        f,
        "float",
        "SCALER_MEAN",
        scaler.mean_
    )


    # --------------------------------------------------------
    # STANDARD SCALER SCALE
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// STANDARD SCALER - SCALE\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    write_float_array(
        f,
        "float",
        "SCALER_SCALE",
        scaler.scale_
    )


    # --------------------------------------------------------
    # PCA MEAN
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// PCA MEAN\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    write_float_array(
        f,
        "float",
        "PCA_MEAN",
        pca.mean_
    )


    # --------------------------------------------------------
    # PCA COMPONENTS
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// PCA COMPONENTS\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    f.write(
        f"const float PCA_COMPONENTS"
        f"[{pca.n_components_}]"
        f"[{pca.n_features_in_}] = {{\n"
    )

    for row_index, row in enumerate(
        pca.components_
    ):

        f.write(
            "    {\n"
        )

        for i in range(
            0,
            len(row),
            6
        ):

            chunk = row[
                i:i + 6
            ]

            formatted_values = []

            for value in chunk:

                value = float(value)

                if not np.isfinite(value):

                    value = 0.0

                formatted_values.append(
                    f"{value:.17g}f"
                )

            line = "        "

            line += ", ".join(
                formatted_values
            )

            if (
                i + 6
                < len(row)
            ):

                line += ","

            f.write(
                line + "\n"
            )

        f.write(
            "    }"
        )

        if (
            row_index
            < len(pca.components_) - 1
        ):

            f.write(",")

        f.write(
            "\n"
        )

    f.write(
        "};\n\n"
    )


    # --------------------------------------------------------
    # DECISION TREE - LEFT CHILD
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// DECISION TREE - LEFT CHILD\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    write_int_array(
        f,
        "int32_t",
        "TREE_LEFT",
        children_left
    )


    # --------------------------------------------------------
    # DECISION TREE - RIGHT CHILD
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// DECISION TREE - RIGHT CHILD\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    write_int_array(
        f,
        "int32_t",
        "TREE_RIGHT",
        children_right
    )


    # --------------------------------------------------------
    # DECISION TREE - FEATURE
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// DECISION TREE - FEATURE INDEX\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    write_int_array(
        f,
        "int32_t",
        "TREE_FEATURE",
        tree_features
    )


    # --------------------------------------------------------
    # DECISION TREE - THRESHOLD
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// DECISION TREE - THRESHOLD\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    write_float_array(
        f,
        "float",
        "TREE_THRESHOLD",
        threshold
    )


    # --------------------------------------------------------
    # DECISION TREE - CLASS
    # --------------------------------------------------------

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// DECISION TREE - PREDICTED CLASS\n"
    )

    f.write(
        "// ============================================================\n\n"
    )

    write_int_array(
        f,
        "int32_t",
        "TREE_CLASS",
        tree_classes
    )


    # --------------------------------------------------------
    # STANDARDIZATION FUNCTION
    # --------------------------------------------------------

    f.write(
r"""
// ============================================================
// STANDARD SCALER
// ============================================================

void standardize_features(
    const float raw_features[NUM_FEATURES],
    float scaled_features[NUM_FEATURES]
)
{
    for (
        int i = 0;
        i < NUM_FEATURES;
        i++
    )
    {
        scaled_features[i] =
            (
                raw_features[i]
                - SCALER_MEAN[i]
            )
            /
            SCALER_SCALE[i];
    }
}


// ============================================================
// PCA TRANSFORMATION
// ============================================================

void apply_pca(
    const float scaled_features[NUM_FEATURES],
    float pca_features[NUM_COMPONENTS]
)
{
    for (
        int component = 0;
        component < NUM_COMPONENTS;
        component++
    )
    {
        float sum = 0.0f;

        for (
            int feature_index = 0;
            feature_index < NUM_FEATURES;
            feature_index++
        )
        {
            sum +=
                (
                    scaled_features[feature_index]
                    - PCA_MEAN[feature_index]
                )
                *
                PCA_COMPONENTS[
                    component
                ][
                    feature_index
                ];
        }

        pca_features[component] = sum;
    }
}


// ============================================================
// DECISION TREE PREDICTION
// ============================================================

int decision_tree_predict(
    const float pca_features[NUM_COMPONENTS]
)
{
    int node = 0;

    while (
        TREE_LEFT[node] != -1
        &&
        TREE_RIGHT[node] != -1
    )
    {
        int feature_index =
            TREE_FEATURE[node];

        if (
            pca_features[feature_index]
            <= TREE_THRESHOLD[node]
        )
        {
            node =
                TREE_LEFT[node];
        }
        else
        {
            node =
                TREE_RIGHT[node];
        }
    }

    return TREE_CLASS[node];
}


// ============================================================
// COMPLETE TINYML INFERENCE
// ============================================================

int classify_flow(
    const float raw_features[NUM_FEATURES]
)
{
    float scaled_features[NUM_FEATURES];

    float pca_features[NUM_COMPONENTS];


    // --------------------------------------------------------
    // STEP 1: STANDARDIZATION
    // --------------------------------------------------------

    standardize_features(
        raw_features,
        scaled_features
    );


    // --------------------------------------------------------
    // STEP 2: PCA
    // --------------------------------------------------------

    apply_pca(
        scaled_features,
        pca_features
    );


    // --------------------------------------------------------
    // STEP 3: DECISION TREE
    // --------------------------------------------------------

    return decision_tree_predict(
        pca_features
    );
}


// ============================================================
// CLASS LABEL
// ============================================================

const char* get_prediction_label(
    int prediction
)
{
    if (prediction == 1)
    {
        return "Attack";
    }

    return "Normal";
}


#endif
"""
    )


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 10 EXPORT COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nGenerated file:"
)

print(
    OUTPUT_FILE
)

print(
    "\nModel:"
)

print(
    "  Input features :",
    len(feature_names)
)

print(
    "  PCA components :",
    pca.n_components_
)

print(
    "  Tree nodes     :",
    node_count
)

print(
    "  Tree depth     :",
    tree_model.get_depth()
)

print(
    "\nClass mapping:"
)

print(
    "  0 = Normal"
)

print(
    "  1 = Attack"
)

print(
    "\nESP32 pipeline:"
)

print(
    "  79 raw features"
)

print(
    "       ↓"
)

print(
    "  StandardScaler"
)

print(
    "       ↓"
)

print(
    "  PCA (10 components)"
)

print(
    "       ↓"
)

print(
    "  Decision Tree"
)

print(
    "       ↓"
)

print(
    "  Normal / Attack"
)

print(
    "\n" + "=" * 70
)