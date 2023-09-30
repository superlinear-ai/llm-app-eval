"""Test llm-app-eval."""

import llm_app_eval


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(llm_app_eval.__name__, str)
