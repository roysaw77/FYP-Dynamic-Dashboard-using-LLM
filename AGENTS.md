# AGENTS.md

Onboarding guide for AI coding agents working in this repository.
**Read this file first, then read the three source files in full:** `src/dashboard.py`, `src/analysis.py`, `src/main.py`. They are the whole application; everything else is data, artifacts, or dependencies.

---

## 1. What this project is

A Final Year Project (FYP) **dynamic dashboard** built around **text-to-SQL over uploaded CSV files**.

Two LLM stages sit on top of a plain Streamlit UI:

| Stage | Stack | File | Question it answers |
| --- | --- | --- | --- |
| Generate | PandasAI 3.0 + `pandasai_litellm` + NVIDIA NIM | `src/dashboard.py` | "What is the answer / chart, and what SQL produced it?" |
| Judge | LiteLLM `completion()` on the same model | `src/dashboard.py` | "Was that a correct answer?" (TRUE / FALSE) |
| Interpret | AutoGen 0.2.x multi-agent group chat | `src/analysis.py` | "What does this mean, and what should the user do?" |

The evaluation design is deliberate: the app does not trust the generated answer. It scrapes the Python code PandasAI executed out of `pandasai.log`, extracts the embedded `sql_query`, and asks a second LLM call to grade the pair. That TRUE/FALSE label plus wall-clock latency is the benchmark signal used in the thesis.

---

## 2. Non-negotiables

1. **Python 3.11 only.** `pandasai-litellm` requires `>=3.10,<3.12`. Use the existing interpreter: `venv\Scripts\python.exe`. Do not `pip install` into a global Python.
2. **Never print, log, echo, cat, commit, or copy `NVIDIA_API_KEY`.** It lives in `.env` (gitignored, ~70 chars) or in Streamlit secrets on Cloud. If you need to verify it exists, print its length or a boolean, never the value.
3. **Always run from the repository root.** See invariant D — running from `src/` corrupts both the log scraping and the chart paths, and has already happened once (there is a stray `src/pandasai.log` and `src/exports/` as evidence).
4. **Do not change the public signatures** of `run_dashboard_analysis()` or `render_agent_analysis()`. `src/main.py` and the demo/README screenshots depend on them. Extend with keyword-only args that have defaults.
5. **Do not add heavy dependencies.** Stick to what `requirements.txt` pins: `pandas`, `streamlit`, `pandasai`, `pandasai-litellm`, `litellm`, `python-dotenv`. AutoGen (`pyautogen`) and matplotlib are already present in the venv but unlisted.
6. **This is graded coursework.** Prefer small, explainable changes over clever ones. Every behaviour change should be describable in one sentence a non-author could defend in a viva.

---

## 3. Environment and commands

All commands assume the repo root as CWD
(`...\FYP-Dynamic-Dashboarding\FYP-Dynamic-Dashboard-using-LLM`).

```powershell
# install / refresh deps
venv\Scripts\python.exe -m pip install -r requirements.txt

# run the main app  (README's "streamlit run main.py" is stale — it is in src/)
venv\Scripts\python.exe -m streamlit run src/main.py

# run the standalone agent-exploration page (browses exports/ PNGs)
venv\Scripts\python.exe -m streamlit run src/analysis.py

# syntax / import smoke check without launching a server
venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); import dashboard, analysis, main"
```

Installed versions this code was written against — do not upgrade casually:

```
pandas 2.3.3      streamlit 1.66.1     pandasai 3.0.0
litellm 1.106.3   pandasai-litellm 0.0.1   pyautogen 0.2.35
numpy 1.26.4      matplotlib 3.7.5    duckdb 1.5.6
```

`pandasai` 3.x has a different API from the widely-documented 2.x. When searching the web or recalling examples, verify against the installed package under `venv/Lib/site-packages/pandasai/`, not memory.

---

## 4. Architecture

### Execution flow

```
main.py
  ├─ st.file_uploader ─► session_state["dataframes"]   (keyed by name+size)
  ├─ selectbox(model) ─► nvidia_nim/<model-id>
  └─ "Run" button
        │
        ▼
dashboard.run_dashboard_analysis(question, dataframes, model_id, api_key)
  1. clear_log("pandasai.log")
  2. generate_Response() ──► pai.config.set({llm: LiteLLM(...)}); pai.chat(q, *dfs)
  3. extract_Generated_code("pandasai.log") ──► {"sql_query", "full_code"}
  4. validation_Response()  ──► litellm.completion(...) ──► "TRUE" | "FALSE" | None
  5. returns dict  ▼
main.py renders: Response / Generated Code / Correctness / Response time
        │
        ▼
analysis.render_agent_analysis(question, result, api_key)
  ├─ if chart: _describe_chart_with_vision(png) ──► text transcription
  └─ run_autogen_discussion(topic, context, ...)
        GroupChat: Coordinator(UserProxy) → ChartAnalyst → DataInterpreter
                   → Critic → Summarizer, round_robin, max_round=N
```

### The result dict (the contract between dashboard.py and analysis.py)

```python
{
  "response_obj":   Response,   # PandasAI object; .type == "chart" | "dataframe" | "string" | "number"
  "response_text":  str,        # .value; for plots this is a FILE PATH, not prose
  "response_time":  float,      # seconds, LLM call only
  "generated_code": str,        # scraped from the log, or "No code generated."
  "sql_query":      str,        # regex-extracted from generated_code; may be "" if the LLM never wrote SQL
  "correctness":    str,        # "TRUE" | "FALSE" | "UNKNOWN"
  "is_chart_export": bool,      # currently a string heuristic (see defect 4)
}
```

`analysis.py` consumes only `is_chart_export`, `response_obj`, `response_text`, `generated_code`. If you add a key, add it here and use `.get()` in `analysis.py` so old session state does not crash.

### Module responsibilities

- `src/main.py` — UI and session state only. No LLM logic belongs here. Uploaded frames are cached by `filename:size` so Streamlit reruns do not reparse; a file removed from the uploader is evicted from `dataframes`.
- `src/dashboard.py` — everything about generation, log scraping, and judging. Pure functions plus two orchestrators. Safe to import headlessly (no Streamlit calls at import except a guarded `st.secrets` read).
- `src/analysis.py` — everything about agents, the vision transcription, and their rendering. Contains both the embedded panel (`render_agent_analysis`) and the standalone page (`render_analysis_page`).

---

## 5. Invariants — break these and the app fails silently or loudly

**A. `llm_config` must be a dict, never a bare `config_list` list.**
`autogen.AssistantAgent(llm_config=[{...}])` raises `AssertionError: llm_config must be a dict or False or None` at construction. The correct shape (as in `analysis_test.ipynb`):

```python
llm_config={"config_list": [{"model": ..., "api_key": ..., "base_url": NIM_BASE_URL}],
            "temperature": 0.4, "cache_seed": None}
```

**B. API keys come from the helpers.** Use `dashboard.get_api_key()` / `analysis._get_api_key()`, which fall back to `st.secrets` for Streamlit Cloud. Reading `os.environ["NVIDIA_API_KEY"]` directly raises `KeyError` on Cloud and bypasses the UI-passed key.

**C. Resolve every path from `Path(__file__)`, never from CWD.** PandasAI writes the chart to a *relative* path and returns that relative string. `st.image()` and `open()` must join it onto the project root:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

**D. PandasAI does not use the log path you think it does.** `pandasai.helpers.logger.Logger` calls `find_closest("pandasai.log")`, which walks *up* from CWD and stops at the first directory containing `pandasai.log`, `pyproject.toml`, `setup.py`, or `requirements.txt`. From the repo root that is `./pandasai.log` — matching `clear_log()` and `extract_Generated_code()`. Run from `src/` (or any subdir holding a stale copy) and PandasAI appends to a different file than the scraper reads: `generated_code` becomes `"No code generated."` and `correctness` degrades to `FALSE`, with no error shown. This is the single most likely "mystery" bug you will be asked to debug.

**E. A chart answer is a path string, and its type is `"chart"`.** The generator writes `result = {"type": "plot", ...}`, but `ResponseParser` maps that to `ChartResponse`, whose `.type` is `"chart"` — so test `.type == "chart"`, never `"plot"`. `response_obj.value` is normally `"exports\\charts\\temp_chart_<uuid>.png"`, but the parser also accepts a `dict` or a `data:image/png;base64,...` data URI, so never assume it is a usable filesystem path. Do not assume prose either. Conversely, `render_agent_analysis` must never feed that path into a text prompt as if it were an answer — transcribe it with the vision model first.

**F. `pai.config` is process-global.** It is mutated on each run and `logging.basicConfig` only takes effect once per Python process, so the log handler is bound at the first PandasAI call in a Streamlit session. Do not assume per-request isolation, and do not parallelise the generation path inside one process.

**G. `extract_Generated_code` is coupled to PandasAI's log format.** It regexes `[INFO] Executing code:` (last occurrence) and then `sql_query = """..."""` / `'...'` / `"..."`. Any PandasAI upgrade requires re-checking `extract_Generated_code` against a fresh log; write a new regex rather than loosening the old one. Note the regex requires the code to *assign* `sql_query`; when the model inlines `execute_sql_query("SELECT ...")` the extraction silently returns `""` — measured at 17% of benchmark rows. PandasAI already exposes the executed code as `response_obj.last_code_executed` (`pandasai/agent/base.py`, `parse(result, code)`), so this whole log dependency is removable — prefer that over another regex.

---

## 6. Known defects (verified, currently unfixed)

Fix these deliberately and one at a time; do not "clean up" around them without confirming with the owner first.

| # | Where | Problem | Impact |
| --- | --- | --- | --- |
| 1 | `analysis.py` → `run_autogen_discussion`, the `chart_analyst` and `data_interpreter` agents | `llm_config=config_list_v` (a list). Confirmed: `AssertionError` on construction. | The agent panel cannot start. Both agents are dead code today. |
| 2 | same function | `config_list_v` reads `os.environ["NVIDIA_API_KEY"]` instead of the passed `api_key`. | `KeyError` on Streamlit Cloud; ignores the UI key. |
| 3 | `dashboard.py` → `clear_log`, `extract_Generated_code`, `run_dashboard_analysis(log_path="pandasai.log")`, and `main.py`'s `st.image(...)` | CWD-relative paths. | Works only when launched from root; see invariant D and the `src/exports/...` double-prefix crash recorded in `analysis_test.ipynb`. |
| 4 | `dashboard.py` → `is_chart_export` | `"export" in response_text` string sniff. | True for any answer containing the word "export"; false for absolute temp paths or data-URI charts. `response_obj.type == "chart"` is the reliable signal. |
| 5 | `dashboard.py` → `run_dashboard_analysis` → `validation_Response` | The judge receives `response_text`, which for charts is a filename. | Chart correctness is graded blind — likely the weakest part of the benchmark and the easiest thing a reviewer will attack. |
| 6 | `analysis.py` → `_render_message` | Agent text is interpolated into HTML rendered with `unsafe_allow_html=True`, unescaped. | A `<` breaks layout; `<script>` is XSS. Escape, then re-apply whitespace handling. |

Lower priority: `DEFAULT_API_KEY` is snapshotted at import time; `exports/charts/temp_chart_*.png` accumulate unbounded (~30 already) and the standalone page lists them all; duplicate upload filenames silently overwrite in session state; `pd.read_csv` has no error handling; `README.md` still says `streamlit run main.py` and `tests/` does not exist.

---

## 7. How to verify a change (no test suite exists)

There is no `tests/` directory and no CI. Anything you change must be proven manually. Minimum bar for the generation path, using `dataset/cc_clinic_level.csv`:

1. Launch from root, upload that CSV, select a model, ask *"Plot top 3 highest clinic revenue"*.
2. **Response** shows an image, not a path string.
3. **Generated Code** contains a real `sql_query = """SELECT ..."""` — if it says `No code generated.`, you have hit invariant D.
4. **Correctness** shows `TRUE`/`FALSE`/`UNKNOWN`, never blank, and **Response time** is a number.
5. Ask a non-chart question about the same file (e.g. *"Which state has the most patients?"*) and confirm the text path still works.
6. Only then open the agent panel and confirm all four agents speak in order and the run terminates on `TERMINATE`.

For `analysis.py` alone: `venv\Scripts\python.exe -m streamlit run src/analysis.py`, pick any PNG under `exports/`, start a discussion.

Cheap static checks before handing back:

```powershell
venv\Scripts\python.exe -m py_compile src\main.py src\dashboard.py src\analysis.py
venv\Scripts\python.exe -c "import ast,sys; [ast.parse(open(f,encoding='utf-8').read()) for f in ['src/main.py','src/dashboard.py','src/analysis.py']]; print('ok')"
```

Each full run costs real API quota and ~10–40 s. Do not loop on live calls to test cosmetic changes; use `py_compile` and small scripts that exercise pure helpers (`extract_Generated_code`, `normalize_text`) against the existing `pandasai.log`.

---

## 8. Common recipes

**Add a model.** Two independent lists, both must be updated: `MODELS` in `dashboard.py` (key is the `nvidia_nim/`-prefixed LiteLLM id) and `AVAILABLE_MODELS` in `analysis.py` (bare NIM id, used by AutoGen). Verify the id exists in the NVIDIA NIM catalogue before adding — several entries here are exotic and may 404.

**Change the judge prompt.** `validation_Response` in `dashboard.py`. Keep the "respond with ONLY one word: TRUE or FALSE" instruction and the `.strip().upper()` parse, or `correctness` becomes multi-line prose and the benchmark table breaks. If you want the reasoning, add a second field instead of relaxing the output contract.

**Add or re-role an agent.** `run_autogen_discussion` in `analysis.py`: append to the `agents=[...]` list, add a matching entry to `AGENT_STYLES` in the same file (otherwise it renders as a grey `🤖` default), and keep `allow_repeat_speaker=False`.

**Change the UI panel.** `render_agent_analysis` — keep it callable on every rerun with a stored result, and keep Streamlit widget `key=` values unique; duplicate keys silently collide across reruns.

---

## 9. Conventions

- Python 3.11, 4-space indent, `snake_case`, type hints on public functions, `Optional[...]`/`Dict[str, Any]` style (the codebase does not use `X | None`).
- Module-level docstrings describe *why*; section banners use `# ── Title ──────` box-drawing comments. Match that visual style when editing existing files.
- Streamlit UI strings are user-facing and intentionally emoji-decorated (🖼️ 🤖 💬 ⏱). Keep the tone; do not strip emoji from labels.
- No f-string SQL assembly, no `eval`, no `exec` on model output. PandasAI already executes LLM-generated Python — do not widen that surface.
- `.env` and `skills-lock.json` are gitignored. `pandasai.log` is a runtime artifact (currently tracked and dirty) — never stage log churn in a commit.

---

## 10. Ask before doing

- Renaming/moving `src/*.py`, or splitting modules. The thesis report references these filenames.
- Changing the result-dict keys or the TRUE/FALSE judge contract (benchmark CSVs in `exports/` were produced under it).
- Committing anything, rewriting history, or touching `dataset/*.sqlite`.
- Adding a paid API call path, a new provider, or anything that needs a key beyond `NVIDIA_API_KEY`.
- Deleting the accumulated `exports/charts/temp_chart_*.png` files — some may be referenced in the write-up.
