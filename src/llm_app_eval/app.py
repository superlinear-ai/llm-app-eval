"""Streamlit app."""

import json
import os
from importlib.metadata import version

import numpy as np
import pandas as pd
import streamlit as st
from eval_properties import properties

st.title(f"llm-app-eval v{version('llm-app-eval')}")  # type: ignore[no-untyped-call]


TEST_SET_FOLDER = "src/llm_app_eval/data/test_cases"
EVAL_FOLDER = "src/llm_app_eval/data/eval_results"
EVAL_RUNS = ["20231001_175828"]

# Load all the test cases JSON files
test_cases = {}  # type: ignore
for test_case in os.listdir(TEST_SET_FOLDER):
    test_case_path = os.path.join(TEST_SET_FOLDER, test_case)
    with open(test_case_path) as f:
        test_cases[test_case] = json.load(f)

# Load all the evaluation results JSON files
eval_results = {}  # type: ignore
for eval_run in EVAL_RUNS:
    eval_results[eval_run] = {}
    eval_run_folder = os.path.join(EVAL_FOLDER, eval_run)
    for eval_file in os.listdir(eval_run_folder):
        eval_file_path = os.path.join(eval_run_folder, eval_file)
        with open(eval_file_path) as f:
            eval_results[eval_run][eval_file] = json.load(f)

# Build a matrix for each evaluation run
# Each row is a test case. Each column is a property.
eval_matrices = {}  # type: ignore
for eval_run in EVAL_RUNS:
    eval_matrices[eval_run] = np.zeros((len(test_cases), len(properties)))
    for test_case_idx, test_case in enumerate(test_cases):
        for property_idx, prop in enumerate(properties):
            r = eval_results[eval_run][test_case]
            for property_result in r["property_results"]:
                if property_result["property_name"] == prop.property_name:
                    eval_matrices[eval_run][test_case_idx, property_idx] = property_result[
                        "pass_fail"
                    ]
                    break
    # Turn the matrix into a dataframe
    eval_matrices[eval_run] = pd.DataFrame(
        eval_matrices[eval_run],
        columns=[prop.property_name for prop in properties],
        index=list(test_cases),
    )

st.write(eval_matrices[eval_run])

# Select a specific test case
test_case = st.selectbox("Test case", list(test_cases.keys()))  # type: ignore

# Select a specific evaluation run
eval_run = st.selectbox("Evaluation run", EVAL_RUNS)  # type: ignore

# Show the test case input
st.markdown("**Test case input:**")
st.write(test_cases[test_case]["test_input"]["question"])
# Show the reference_output, historical_output, and historical_feedback, if available
if test_cases[test_case]["reference_output"]:
    st.markdown("**Reference output:**")
    st.write(test_cases[test_case]["reference_output"]["answer"])
if test_cases[test_case]["historical_output"]:
    st.markdown("**Historical output:**")
    st.write(test_cases[test_case]["historical_output"]["answer"])
if test_cases[test_case]["historical_feedback"]:
    st.markdown("**Historical feedback:**")
    st.write(test_cases[test_case]["historical_feedback"])

# Show the model output
st.markdown("**Model response:**")
st.write(eval_results[eval_run][test_case]["output"]["answer"])

# Show the evaluation results
st.markdown("**Evaluation results:**")
# Loop over the properties
for prop in properties:
    # Loop over the evaluation runs
    for eval_run in EVAL_RUNS:
        # Loop over the evaluation results
        for property_result in eval_results[eval_run][test_case]["property_results"]:
            # If the property name matches the current property, show the result
            if property_result["property_name"] == prop.property_name:
                st.write(f"{prop.property_name}: {'✅' if property_result['pass_fail'] else '❌'}")
                st.write(property_result["feedback"])
