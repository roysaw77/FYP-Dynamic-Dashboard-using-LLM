import io
import re
from pathlib import Path
import time

import pandas as pd
import streamlit as st

import pandasai as pai
from pandasai_litellm.litellm import LiteLLM
from litellm import completion



DEFAULT_MODEL = "nvidia_nim/meta/llama-3.1-405b-instruct"
DEFAULT_API_KEY = "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"

st.set_page_config(page_title="PandasAI CSV", page_icon="💬", layout="wide")
st.title("PandasAI CSV")


def normalize_text(value: str) -> str:
	return re.sub(r"\s+", " ", str(value).strip())


def extract_Generated_code(log_path: str = "pandasai.log"):
    """
    Extract the full generated code from PandasAI log.
    
    Returns:
        dict: {"sql_query": str, "full_code": str}
    """
    from pathlib import Path
    
    log_file = Path(log_path)
    if not log_file.exists():
        return {"sql_query": "", "full_code": ""}
    
    text = log_file.read_text(encoding="utf-8", errors="replace")
    
    # --- Extract full generated code from "Executing code:" section ---
    full_code = ""
    # Pattern to find code after "Executing code:" until next log entry or end
    exec_pattern = r'\[INFO\] Executing code:\s*(.+?)(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[INFO\]|$)'
    exec_matches = re.findall(exec_pattern, text, re.DOTALL)
    if exec_matches:
        full_code = exec_matches[-1].strip()
    
    # --- Extract SQL query from the full code ---
    sql_query = ""
    # Pattern for triple-quoted multi-line SQL
    multi_line_pattern = r'sql_query\s*=\s*"""(.+?)"""'
    single_quote_pattern = r"sql_query\s*=\s*'([^']+)'"
    double_quote_pattern = r'sql_query\s*=\s*"([^"]+)"'

    # Search in full_code first, then in entire text
    search_text = full_code if full_code else text
    
    matches = re.findall(multi_line_pattern, search_text, re.DOTALL)
    if not matches:
        matches = re.findall(single_quote_pattern, search_text)
    if not matches:
        matches = re.findall(double_quote_pattern, search_text)
    if matches:
        sql_query = matches[-1].strip()
    
    return {"sql_query": sql_query, "full_code": full_code}


def clear_log(log_path: str = "pandasai.log"):
    """Clear the PandasAI log file after extraction."""
    from pathlib import Path
    log_file = Path(log_path)
    if log_file.exists():
        log_file.write_text("", encoding="utf-8")


def generate_Response(question: str, dataframes: list, model_id: str = None, api_key: str = None):
    """
    Generate a response using PandasAI for the given question.
    
    Args:
        question: The question to ask
        dataframes: List of PandasAI DataFrames to query
        model_id: The LLM model identifier
        api_key: API key for the LLM
    
    Returns:
        tuple: (response, elapsed_time)
    """
    default_api_key = "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"
    default_model = "nvidia_nim/meta/llama-3.1-8b-instruct"
    
    # Configure the LLM
    llm = LiteLLM(
        model=model_id or default_model,
        api_key=api_key or default_api_key,
        stream=False,
    )
    pai.config.set({"llm": llm, "save_charts": False,"verbose": False})
    
    start_time = time.time()
    response = pai.chat(question, *dataframes)
    elapsed_time = time.time() - start_time
    
    return response, elapsed_time


def validation_Response(query: str, question: str, response=None, answer=None, api_key: str = None) -> bool:
    if not query:
        return False
    
    try:
        validation = completion(
            model="nvidia_nim/meta/llama-3.1-405b-instruct",
            messages = [{
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
                            """
                        }],
            api_key=api_key or "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"
            
        )
        return validation.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
        return None

if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "file_signatures" not in st.session_state:
    st.session_state.file_signatures = {}

uploaded_files = st.file_uploader(
    "Upload CSV files", type=["csv"], accept_multiple_files=True, key="csv_uploader_main"
)

uploaded_names = []
if uploaded_files:
    for file in uploaded_files:
        uploaded_names.append(file.name)
        signature = f"{file.name}:{file.size}"
        if st.session_state.file_signatures.get(file.name) != signature:
            data = io.BytesIO(file.getvalue())
            st.session_state.dataframes[file.name] = pd.read_csv(data)
            st.session_state.file_signatures[file.name] = signature

    for name in list(st.session_state.dataframes.keys()):
        if name not in uploaded_names:
            st.session_state.dataframes.pop(name, None)
            st.session_state.file_signatures.pop(name, None)

selected_files = st.multiselect(
    "Datasets for PandasAI",
    options=uploaded_names,
    default=uploaded_names,
)

api_key = st.text_input("API Key", type="password", value=DEFAULT_API_KEY)
model_id = st.text_input("Model", value=DEFAULT_MODEL)
question = st.text_input("Question")


if st.button("Run", type="primary"):
    if not st.session_state.dataframes:
        st.error("Upload at least one CSV file.")
    elif not selected_files:
        st.error("Select at least one dataset.")
    elif not question.strip():
        st.error("Enter a question.")
    elif not api_key.strip():
        st.error("Enter API key.")
    else:
        selected_dataframes = [st.session_state.dataframes[name] for name in selected_files]
        try:
            with st.spinner("Running PandasAI and validating result..."):
                clear_log()
                response_obj, response_time = generate_Response(
                    question, selected_dataframes, model_id.strip(), api_key.strip()
                )
                response_text = response_obj.value if hasattr(response_obj, "value") else str(response_obj)
                generated = extract_Generated_code()
                sql_code = generated.get("sql_query", "")
                code_to_validate = generated.get("full_code", "") or sql_code

                judge = validation_Response(
                    code_to_validate,
                    question,
                    response=response_text,
                    
                    api_key=api_key.strip(),
                )
                correctness = str(judge).strip().upper()

            st.subheader("Response")
            if("export" in response_text):
                st.image(response_obj.value)
            else:
                st.code(response_text)
            st.subheader("SQL Code")
            st.code(sql_code if sql_code else "")
            st.subheader("Correctness")
            st.code(correctness)
            st.subheader("Response time")
            st.code(f"{response_time:.2f} seconds")
        except Exception as exc:
            st.error(str(exc))
