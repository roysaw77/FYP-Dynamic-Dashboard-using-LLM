import telebot
import ollama
import pandas as pd
import io
import os
import time
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM



TELEGRAM_API_TOKEN = "8264826455:AAFc_XinYgj_EUn9Z6I51E2TKCDMUKNVRRw"
# Initialize LiteLLM with your OpenAI model
CHARTS_DIR = "exports/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

llm = LiteLLM(model="ollama/qwen3:8b", api_key="http://localhost:11434/api")
config = {
    "llm": llm,
    # "save_charts": True,
    # "save_charts_path": CHARTS_DIR,
    # "open_charts": False,
    # "enable_cache": False,
}
pai.config.set(config)
MODEL = "llama3:latest"
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


selected_indices = []
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
    
    start_time = time.time()
    try:
        selected_dfs = [csvarr[i] for i in selected_indices]
        response = pai.chat(question, *selected_dfs)
    except Exception as exc:
        bot.reply_to(message, f"Error: {exc}")
        return
    
    elapsed_time = time.time() - start_time
    
    result_message = f"{str(response.value)}\n\n"
    result_message += f"🤖 Model: {llm.model}\n"
    result_message += f"⏱️ Processing time: {elapsed_time:.2f}s"
    
    bot.reply_to(message, result_message)

    


print("Bot is running...")
bot.polling()
#how plot output into telegram

#fine tuning