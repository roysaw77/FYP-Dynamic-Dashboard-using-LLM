import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from litellm import completion
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

# Load environment variables from project root .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


def get_api_key() -> str:
    """Retrieve the NVIDIA API key from environment variables or Streamlit secrets."""
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
                key = st.secrets["NVIDIA_API_KEY"]
        except Exception:
            pass
    return (key or "").strip()


DEFAULT_MODEL = "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_API_KEY = get_api_key()

MODELS = {
    "nvidia_nim/meta/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "nvidia_nim/google/gemma-2-27b-it": "Gemma-2-27B",
    "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b": "Nemotron-3.5-lightning-30B",
    "nvidia_nim/meta/llama-3.1-405b-instruct": "Llama-3.1-405B",
    "nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512": "Mistral-Large-3-675B",
}


def normalize_text(value: str) -> str:
    """Strip and collapse irregular whitespace into single spaces."""
    return re.sub(r"\s+", " ", str(value).strip())


def extract_Generated_code(log_path: str = "pandasai.log") -> Dict[str, str]:
    """
    Extract the full generated code and SQL query from PandasAI log file.

    Returns:
        dict: {"sql_query": str, "full_code": str}
    """
    log_file = Path(log_path)
    if not log_file.exists():
        return {"sql_query": "", "full_code": ""}

    text = log_file.read_text(encoding="utf-8", errors="replace")

    # Extract full generated code from "Executing code:" section
    full_code = ""
    exec_pattern = r'\[INFO\] Executing code:\s*(.+?)(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[INFO\]|$)'
    exec_matches = re.findall(exec_pattern, text, re.DOTALL)
    if exec_matches:
        full_code = exec_matches[-1].strip()

    # Extract SQL query from the full code or entire log text
    sql_query = ""
    multi_line_pattern = r'sql_query\s*=\s*"""(.+?)"""'
    single_quote_pattern = r"sql_query\s*=\s*'([^']+)'"
    double_quote_pattern = r'sql_query\s*=\s*"([^"]+)"'

    search_text = full_code if full_code else text
    matches = re.findall(multi_line_pattern, search_text, re.DOTALL)
    if not matches:
        matches = re.findall(single_quote_pattern, search_text)
    if not matches:
        matches = re.findall(double_quote_pattern, search_text)
    if matches:
        sql_query = matches[-1].strip()

    return {"sql_query": sql_query, "full_code": full_code}


def clear_log(log_path: str = "pandasai.log") -> None:
    """Clear the PandasAI log file before starting a new run."""
    log_file = Path(log_path)
    if log_file.exists():
        log_file.write_text("", encoding="utf-8")


def generate_Response(
    question: str,
    dataframes: List[pd.DataFrame],
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Any, float]:
    """
    Generate a response using PandasAI for the given question and DataFrames.

    Returns:
        tuple: (response_object, elapsed_time_seconds)
    """
    llm = LiteLLM(
        model=model_id or DEFAULT_MODEL,
        api_key=api_key or DEFAULT_API_KEY,
        stream=False,
    )
    pai.config.set({"llm": llm, "save_charts": False, "verbose": False})

    start_time = time.time()
    response = pai.chat(question, *dataframes)
    elapsed_time = time.time() - start_time

    return response, elapsed_time


def validation_Response(
    query: str,
    question: str,
    response: Any = None,
    answer: Any = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Evaluate whether the generated query and result accurately answer the question.

    Returns:
        str: "TRUE", "FALSE", or None on error.
    """
    if not query:
        return "FALSE"

    try:
        validation = completion(
            model=model_id or DEFAULT_MODEL,
            messages=[{
                "role": "user",
                "content": f"""
You are an execution accuracy evaluator for a Text-to-SQL system.

Question:
{question}

Generated SQL:
{query}

Execution Result:
{response}

Task:
Determine whether the SQL query and its execution result correctly answer the given question.

Evaluation Guidelines:
- Check if the SQL query logically matches the intent of the question.
- Verify whether the execution result is consistent with what the question is asking.
- Ensure the result is complete, relevant, and not misleading.
- If the result is empty or irrelevant when it should contain data, return FALSE.
- If the result contains incorrect aggregation, filtering, or columns, return FALSE.
- If the result is a chart or visualization, evaluate whether it correctly represents the intended data and relationships.

Important:
- You do NOT have access to the ground truth answer.
- Base your judgment only on the question, SQL query, and execution result.

Respond with ONLY one word: TRUE or FALSE.
                """.strip(),
            }],
            api_key=api_key or DEFAULT_API_KEY,
        )
        return validation.choices[0].message.content.strip()
    except Exception as e:
        print(f"Validation error: {e}")
        return None


def run_dashboard_analysis(
    question: str,
    dataframes: List[pd.DataFrame],
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    log_path: str = "pandasai.log",
) -> Dict[str, Any]:
    """
    High-level orchestrator that clears logs, invokes PandasAI, extracts code,
    and runs the accuracy evaluation judge.
    """
    clear_log(log_path)

    response_obj, response_time = generate_Response(
        question=question,
        dataframes=dataframes,
        model_id=model_id,
        api_key=api_key,
    )

    response_text = response_obj.value if hasattr(response_obj, "value") else str(response_obj)
    generated = extract_Generated_code(log_path)
    sql_code = generated.get("sql_query", "")
    full_code = generated.get("full_code", "")
    code_to_validate = full_code or sql_code

    judge = validation_Response(
        query=code_to_validate,
        question=question,
        response=response_text,
        model_id=model_id,
        api_key=api_key,
    )
    correctness = str(judge).strip().upper() if judge is not None else "UNKNOWN"

    is_chart_export = isinstance(response_text, str) and ("export" in response_text or response_text.endswith(".png"))

    return {
        "response_obj": response_obj,
        "response_text": response_text,
        "response_time": response_time,
        "generated_code": full_code if full_code else "No code generated.",
        "sql_query": sql_code,
        "correctness": correctness,
        "is_chart_export": is_chart_export,
    }
