"""Streamlit app."""

from importlib.metadata import version

import numpy as np
import streamlit as st
from evaluator import Evaluator
from qa_extraction import load_qa_pairs

st.title(f"llm-app-eval v{version('llm-app-eval')}")  # type: ignore[no-untyped-call]


qa_pairs = load_qa_pairs("src/llm_app_eval/data/question_answer_pairs.csv")
evaluator = Evaluator(llm="gpt-4")

# Shuffle the question and answer pairs
np.random.seed(42)
np.random.shuffle(qa_pairs)
# Display a question and answer pair
if "idx" in st.session_state:
    idx = st.session_state.idx
else:
    idx = 0
    st.session_state.idx = idx
st.write(f"Question {idx + 1} of {len(qa_pairs)}")
qa = qa_pairs[idx]
st.header("Question")
st.write(qa.question)
st.header("Answer")
answer = st.text_input("Answer")
st.header("Reference Answer")
st.write(qa.answer)


eval_button = st.button("Evaluate")
if eval_button:
    result = evaluator.evaluate(qa.question, answer, qa.answer)
    st.write("✅" if result.pass_fail else "❌")
    st.write(result.feedback)
    st.session_state.idx = min(st.session_state.idx + 1, len(qa_pairs) - 1)
else:
    # Display previous and next buttons
    col1, col2, col3 = st.columns(3)
    if col1.button("Previous"):
        st.session_state.idx = max(st.session_state.idx - 1, 0)
    if col2.button("Random"):
        st.session_state.idx = np.random.randint(0, len(qa_pairs))
    if col3.button("Next"):
        st.session_state.idx = min(st.session_state.idx + 1, len(qa_pairs) - 1)
