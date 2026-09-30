from pathlib import Path

import pytest


def test_streamlit_dashboard_app_loads_without_exceptions():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    project_root = Path(__file__).resolve().parents[1]
    app = streamlit_testing.AppTest.from_file(str(project_root / "dashboard" / "app.py"))
    app.run(timeout=60)

    assert len(app.exception) == 0
    assert len(app.radio) == 1
    assert app.radio[0].options == [
        "Overview", "Model Comparison", "8-Dimension Analysis", "Response Efficiency",
        "Task Results", "Evidence & Limitations", "Reports", "Settings",
    ]
    rendered = "\n".join(element.value for element in app.markdown)
    assert "REAL EXPERIMENT" in rendered
    assert "Overall Score Unavailable" in rendered
    assert "Response Efficiency" in rendered
    assert "qwen3:4b" in rendered
    assert "qwen3.5:latest" in rendered
    assert "llama3.2:3b" in rendered
    assert "Complete" in rendered and "Partial" in rendered and "Failed" in rendered
    assert "Model_A" not in rendered
    assert rendered.count('class="kpi-card accent-') == 7
    assert "8-Dimension Comparison" in rendered
    assert "Response Status Distribution" in rendered
    assert "Task Results" in rendered and "P005" in rendered
    comparison_markup = "\n".join(element.value for element in app.markdown)
    assert "<table class=\"comparison-table\">" in comparison_markup
    assert "Correctness" in comparison_markup
    assert "Overall Score" in comparison_markup
    assert "Robustness / Reliability" in comparison_markup
    assert "N/A" in comparison_markup
    assert "ollama / qwen3:4b" in comparison_markup
    assert "ollama / qwen3.5:latest" in comparison_markup
    assert "ollama / llama3.2:3b" in comparison_markup
    assert "DEMO / MOCK" not in rendered

    app.radio[0].set_value("Response Efficiency").run(timeout=60)
    efficiency = app.dataframe[0].value
    assert len(efficiency) == 15
    assert set(efficiency["Task ID"]) == {"P001", "P002", "P003", "P004", "P005"}
    assert set(efficiency["Model"]) == {"qwen3:4b", "qwen3.5:latest", "llama3.2:3b"}

    for page in app.radio[0].options:
        app.radio[0].set_value(page).run(timeout=60)
        assert len(app.exception) == 0, f"Dashboard page raised an exception: {page}"
