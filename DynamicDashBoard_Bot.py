import telebot
import pandas as pd
import os
import time
import re
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM
from pathlib import Path
from litellm import completion

def generate_Response(question: str, dataframes: list, model_id: str = None, api_key: str ='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'):
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
    default_model = "nvidia_nim/google/llama3-70b-instruct"
    
    # Configure the LLM
    llm = LiteLLM(
        model=model_id or default_model,
        api_key=api_key or default_api_key,
        stream=False,
    )
    pai.config.set({"llm": llm, "save_charts": False})
    
    start_time = time.time()
    response = pai.chat(question, *dataframes)
    elapsed_time = time.time() - start_time
    
    return response, elapsed_time

def extract_Sql_query(log_path: str = "pandasai.log") -> str:
    """
    Extracts the SQL query from the PandasAI log file.

    Args:
        log_path (str): Path to the pandasai.log file.

    Returns:
        str: The extracted SQL query, or empty string if not found.
    """
    from pathlib import Path
    
    log_file = Path(log_path)
    if not log_file.exists():
        return ""
    
    text = log_file.read_text(encoding="utf-8")
    
    # Pattern for triple-quoted multi-line SQL
    multi_line_pattern = r'sql_query\s*=\s*"""(.+?)"""'
    # Pattern for single-quoted SQL
    single_quote_pattern = r"sql_query\s*=\s*'([^']+)'"
    # Pattern for double-quoted SQL
    double_quote_pattern = r'sql_query\s*=\s*"([^"]+)"'

    # Try multi-line first, then single/double quotes
    matches = re.findall(multi_line_pattern, text, re.DOTALL)
    if not matches:
        matches = re.findall(single_quote_pattern, text)
    if not matches:
        matches = re.findall(double_quote_pattern, text)

    if matches:
        return matches[-1].strip()  # Return the latest one, cleaned
    return ""


def clear_log(log_path: str = "pandasai.log"):
    """Clear the PandasAI log file after extraction."""
    log_file = Path(log_path)
    if log_file.exists():
        log_file.write_text("", encoding="utf-8")

def validation_Response(sql_query: str, question: str, api_key: str = None) -> bool:
    """
    Validates if the SQL query correctly answers the given question using LLM.

    Args:
        sql_query (str): The SQL query to validate.
        question (str): The original question the query should answer.
        api_key (str): API key for the validation LLM.

    Returns:
        bool: True if the query is correct, False otherwise, None if validation failed.
    """
    if not sql_query:
        return False
    
    try:
        validation = completion(
            model="nvidia_nim/google/gemma-2-27b-it",
            messages=[{
                "role": "user", 
                "content": f"You just answer me true or false do not explain: does this query: {sql_query} correctly answer the question: {question}"
            }],
            api_key=api_key or "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",
        )
        return "true" in validation.choices[0].message.content.lower()
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
        return None

def LLM_explaination(sql_query: str, api_key: str = None) -> str:
    """
    Get an explanation of the SQL query using LLM.

    Args:
        sql_query (str): The SQL query to explain.
        api_key (str): API key for the explanation LLM.

    Returns:
        str: Explanation of the SQL query.
    """
    try:
        explanation = completion(
            model="nvidia_nim/google/gemma-2-27b-it",
            messages=[{
                "role": "user", 
                "content": f"Explain this SQL query in simple terms: {sql_query}"
                }],
                api_key=api_key or "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",
            )
        return explanation.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Explanation error: {e}")
        return "Could not generate explanation."

# telegram bot
TELEGRAM_API_TOKEN = "8264826455:AAFc_XinYgj_EUn9Z6I51E2TKCDMUKNVRRw"
# Initialize LiteLLM with your OpenAI model
CHARTS_DIR = "exports/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

llm=LiteLLM(
    model="nvidia_nim/google/llama3-70b-instruct",
    api_key="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",
    stream=False,
)
pai.config.set({"llm": llm, "save_charts": True, "charts_dir": CHARTS_DIR})#default

df_clinic_level = pd.read_csv("cc_clinic_level.csv")
df_doctor = pd.read_csv("cc_doctor.csv")
df_hourly = pd.read_csv("cc_hourly.csv")
df_patient = pd.read_csv("cc_patient.csv")

df1 = pai.DataFrame(df_clinic_level)
df2 = pai.DataFrame(df_doctor)
df3 = pai.DataFrame(df_hourly)
df4 = pai.DataFrame(df_patient)

csvarr = [df1, df2, df3, df4]
csv_names = ["cc_clinic_level.csv", "cc_doctor.csv", "cc_hourly.csv", "cc_patient.csv"]


selected_indices = ["cc_clinic_level.csv"]
bot = telebot.TeleBot(TELEGRAM_API_TOKEN)


#  read csv file
@bot.message_handler(commands=['list'])
def handle_list(message):
    response = "Available datasets:\n"
    response += "1. cc_clinic_level.csv\n"
    response += "2. cc_doctor.csv\n"
    response += "3. cc_hourly.csv\n"
    response += "4. cc_patient.csv\n"
    bot.reply_to(message, response)

@bot.message_handler(commands=['choose'])
def handle_selectedcsv(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "Usage: /choose <1-4>")
        return

    try:
        idx = int(parts[1]) - 1
    except ValueError:
        bot.reply_to(message, "Please provide a number after /choose.")
        return

    if idx < 0 or idx >= len(csvarr):
        bot.reply_to(message, "Invalid dataset number. Please choose 1-4.")
        return

    global selected_indices
    selected_indices.append(idx)
    names = [csv_names[i] for i in selected_indices]
    bot.reply_to(message, f"Datasets selected: {', '.join(names)}")


@bot.message_handler(commands=['show'])
def handle_show(message):
    if not selected_indices:
        bot.reply_to(message, "No datasets selected.")
        return
    names = [csv_names[i] for i in selected_indices]
    listing = "\n".join(f"{i+1}. {name}" for i, name in enumerate(names))
    bot.reply_to(message, f"Selected datasets:\n{listing}")


@bot.message_handler(commands=['clear'])
def handle_remove(message):
    global selected_indices
    selected_indices = []
    bot.reply_to(message, "Cleared all selected datasets.")
   

@bot.message_handler(commands=['ask'])
def handle_ask(message):
    if not selected_indices:
        bot.reply_to(message, "No datasets selected. Use /choose to select datasets.")
        return
    
    question = message.text[len('/ask'):].strip()
    if not question:
        bot.reply_to(message, "Usage: /ask <your question>")
        return
    
    # Send a "processing" message
    processing_msg = bot.reply_to(message, "Processing your question...")
    
    start_time = time.time()
    try:
        # Clear the log before generating response
        clear_log()
        
        # Generate response using PandasAI
        selected_dfs = [csvarr[i] for i in selected_indices]
        response, elapsed_time = generate_Response(question, *selected_dfs)
        
        # Extract the SQL query from the log
        sql_query = extract_Sql_query()
        
        # Validate the SQL query
        is_valid = validation_Response(sql_query, question)
        
        # Get LLM explanation of the SQL query
        explanation = LLM_explaination(sql_query) if sql_query else "No SQL query generated."
        
    except Exception as exc:
        bot.edit_message_text(f"Error: {exc}", message.chat.id, processing_msg.message_id)
        return
    
    # Build the result message
    result_message = "📊 **Query Results**\n\n"
    
    # 1. Validation Response
    if is_valid is None:
        validation_status = "⚠️ Validation: Could not validate"
    elif is_valid:
        validation_status = "✅ Validation: True"
    else:
        validation_status = "❌ Validation: False"
    result_message += f"{validation_status}\n\n"
    
    # 2. SQL Query
    result_message += f"🔍 **SQL Query:**\n```\n{sql_query if sql_query else 'No SQL query extracted'}\n```\n\n"
    
    # 3. Answer
    result_message += f"📝 **Answer:**\n{str(response.value)}\n\n"
    
    # 4. LLM Explanation
    result_message += f"💡 **Explanation:**\n{explanation}\n\n"
    
    # 5. Processing Time
    result_message += f"⏱️ **Processing time:** {elapsed_time:.2f}s\n"
    result_message += f"🤖 **Model:** {llm.model}"
    
    # Edit the processing message with the final result
    bot.edit_message_text(result_message, message.chat.id, processing_msg.message_id, parse_mode='Markdown')

print("Bot is running...")
bot.polling()
