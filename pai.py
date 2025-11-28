import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

# Initialize LiteLLM with your OpenAI model
llm = LiteLLM(model="ollama/llama3", api_key="http://localhost:11434/api")

# Configure PandasAI to use this LLM
pai.config.set({
    "llm": llm
})

# Load your data
df = pai.read_csv("cc_clinic_level.csv")

response = df.chat("Plot the histogram of clinic showing for each one the revenue. Use different colors for each bar")
print(response)