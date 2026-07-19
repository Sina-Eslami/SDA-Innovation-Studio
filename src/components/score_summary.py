import streamlit.components.v1 as components
from pathlib import Path
import json


def score_summary(
    desirability,
    viability,
    feasibility
):

    avg = (
        desirability +
        viability +
        feasibility
    ) / 3


    data = {
        "desirability": desirability,
        "viability": viability,
        "feasibility": feasibility
    }


    html_path = (
        Path(__file__)
        .parent
        / "score_summary.html"
    )


    html = html_path.read_text()


    html = html.replace(
        "__AVERAGE__",
        f"{avg:.1f}"
    )


    html = html.replace(
        "__DATA__",
        json.dumps(data)
    )


    components.html(
        html,
        height=520,
        scrolling=False
    )