import telebot
import ollama
import pandas as pd
import io
import os
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM
import matplotlib
matplotlib.use('Agg')



# --- CONFIGURATION ---

# WARNING: You posted your token publicly. You should revoke it in BotFather and get a new one!

TELEGRAM_API_TOKEN = "8264826455:AAFc_XinYgj_EUn9Z6I51E2TKCDMUKNVRRw"
# Initialize LiteLLM with your OpenAI model
CHARTS_DIR = "exports/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

llm = LiteLLM(model="ollama/llama3", api_key="http://localhost:11434/api")
config = {
    "llm": llm,
    "save_charts": True,
    "save_charts_path": CHARTS_DIR,
    "open_charts": False,
    "enable_cache": False,
}
pai.config.set(config)
MODEL = "llama3:latest"



PERSONAL_ID = None
bot = telebot.TeleBot(TELEGRAM_API_TOKEN)

@bot.message_handler(commands=['start', 'help'])

def send_welcome(message):
    print(f"User {message.from_user.first_name} (ID: {message.chat.id}) started the bot.")
    bot.reply_to(message, f"Hello! Your ID is {message.chat.id}. Copy this to your script if you want to restrict access.")

def ask_llm(prompt: str) -> str:
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {e}"



@bot.message_handler(func=lambda message: True)
def handle_message(message):
    # 1. Log the chatter
    print(f"Received message from ID {message.chat.id}: {message.text}")
    # 2. Security Check (Optional)
    # Note: message.chat.id is a property, NOT a function message.chat.id()
    if PERSONAL_ID is not None and message.chat.id != PERSONAL_ID:
        bot.reply_to(message, "⛔ Unauthorized user.")
        return 
    # 3. Send "Typing..." status because LLMs are slow
    bot.send_chat_action(message.chat.id, 'typing')
    # 4. Get response
    response = ask_llm(message.text)
    # 5. Reply
    bot.reply_to(message, response)


#  read csv file
@bot.message_handler(content_types=['document'])
def handle_document(message):
    try:
        # 1. Check if it is a CSV
        file_name = message.document.file_name
        if not file_name.endswith('.csv'):
            bot.reply_to(message, "I can only read .csv files right now.")
            return       

        bot.reply_to(message, f"📥 Reading {file_name}...")
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        df = pd.read_csv(io.BytesIO(downloaded_file))
        
      
        pai_df = pai.DataFrame(df)
       
        user_question = message.caption if message.caption else "Please summarize this dataset."
        bot.send_chat_action(message.chat.id, 'typing')

        response = pai_df.chat(user_question)
        
        if isinstance(response, str) and response.endswith('.png') and os.path.exists(response):
            print(f"Chart generated at: {response}")
            with open(response, 'rb') as photo:
                bot.send_photo(message.chat.id, photo, caption="Here is the chart you requested.")
        else:
            # Otherwise, it's just text
            bot.reply_to(message, str(response))


    except Exception as e:
        bot.reply_to(message, f"Error processing file: {e}")

print("Bot is running...")
bot.polling()
#how plot output into telegram