# PPT Content

## Slide 1 — Title
A Multi-Dimensional Quality Evaluation Framework for LLM-Based Intelligent Programming Assistants

## Slide 2 — Background
LLM programming assistants can generate, explain and modify code. Code correctness alone does not capture overall software quality.

## Slide 3 — Problem Statement
Need a systematic evaluation framework covering multiple quality dimensions.

## Slide 4 — Research Gap
Conventional code-generation evaluation often emphasizes functional correctness; this project combines multiple quality dimensions.

## Slide 5 — Objectives
Develop a reproducible framework; evaluate multiple assistants; analyze quality/efficiency trade-offs; incorporate user evaluation.

## Slide 6 — Research Questions
RQ1-RQ5 from research_questions.md.

## Slide 7 — Proposed Architecture
Benchmark -> LLMs -> Evaluation Modules -> Scoring -> Database -> Analytics -> Dashboard.

## Slide 8 — Dataset
HumanEval/MBPP or another documented programming benchmark.

## Slide 9 — Evaluation Dimensions
Correctness, completeness, security, maintainability, explainability, usability, efficiency.

## Slide 10 — Implementation
Python, LLM API, pytest/subprocess, Bandit, Radon, Pandas, SQLite, Plotly, Streamlit.

## Slide 11 — Experimental Procedure
Same tasks + same prompts -> collect responses -> evaluate -> score -> analyze.

## Slide 12 — Dashboard
Show model filters, dimension scores, overall scores, response-time plots and detailed results.

## Slide 13 — Results
Insert actual experimental tables and charts.

## Slide 14 — Discussion
Answer each research question using evidence from the experiment.

## Slide 15 — Limitations
Benchmark size, model/API variability, evaluator subjectivity, security-analysis coverage, scoring assumptions.

## Slide 16 — Conclusion & Future Work
Summarize findings and propose larger benchmarks, more languages, stronger sandboxing and longitudinal evaluation.
