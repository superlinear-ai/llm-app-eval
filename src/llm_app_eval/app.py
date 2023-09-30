"""Streamlit app."""

from importlib.metadata import version

import streamlit as st

st.title(f"llm-app-eval v{version('llm-app-eval')}")  # type: ignore[no-untyped-call]
