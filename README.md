# Dynamic Dashboard using LLM

## Overview

This project is aimed at developing a dynamic dashboard utilizing Large Language Models (LLMs) to provide tailored data visualizations and interactions.

## Research Paper
[thesis (1).pdf](https://github.com/user-attachments/files/27086468/thesis.1.pdf)

## Demo Video
<video controls src="Dynamic Dashboard Demo.mp4" title="Title"></video>

## Key Features
- **LLM-driven dashboard generation**: Automatically generate dashboards based on user input and data context.
- **Data ingestion**: Seamlessly ingest data from various sources for comprehensive analysis.
- **Visualization**: Create rich and interactive visual representations of data.
- **Prompt templates**: Use customizable prompt templates for different use cases.

## Tech Stack 
- Frontend: [Streamlit]
- Framework: Generate Dashboard [Pandasai, LLM]
             Analysis Insight [Autogen, VLM]
- Generate Dashboard Framework
<img width="818" height="420" alt="image" src="https://github.com/user-attachments/assets/2ff4dc74-45d6-4be5-ad2a-b03845c5255b" />

- Multi-Agent Analysis Framework
![alt text](<Multi-Agent Analysis Framework.png>)



## Project Structure
```
project/
├── src/                    # Source files streamlit web page of Dynamic Dashboarding
├── dataset/                # dataset for experiment
├── exports/                # Generated charts and benchmark evaluations
└── tests/                  # experiment test
```

## Webpage
- https://fyp-dynamic-dashboard-using-llm.streamlit.app/

## Setup Instructions

### Prerequisites
- **Python 3.11** (Required: `>=3.10, <3.12` due to `pandasai-litellm`; Python 3.11.9 recommended).
- **NVIDIA NIM API Key**: Required for LLM/VLM generation and AutoGen multi-agent analysis. Obtain an API key at [build.nvidia.com](https://build.nvidia.com).

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/roysaw77/FYP-Dynamic-Dashboard-using-LLM.git
   cd FYP-Dynamic-Dashboard-using-LLM
   ```

2. **Create and activate a Python 3.11 virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3.11 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration
Create a `.env` file in the project root directory:
```env
NVIDIA_API_KEY=nvapi-your_api_key_here
```
*(On Streamlit Community Cloud, set `NVIDIA_API_KEY = "nvapi-..."` in **App Settings > Secrets**).*

## How to Run in Streamlit

> **Important**: Always launch commands from the **repository root** (`FYP-Dynamic-Dashboard-using-LLM`) so that relative paths, chart exports, and log files resolve correctly.

### 1. Launch the Main Application
Start Streamlit from the project root:
```bash
streamlit run src/main.py
```
*(Or explicitly using the virtual environment interpreter: `venv\Scripts\python.exe -m streamlit run src/main.py`)*

### 2. Open in Your Browser
Once started, Streamlit will display the local URL in the terminal (typically `http://localhost:8501`). Open this URL in your web browser.

### 3. Application Pages
The sidebar navigation allows switching between:
- **📊 Dashboard**: Upload CSV datasets, ask questions in natural language, generate text-to-SQL results and charts (powered by PandasAI + NVIDIA NIM), and inspect generated SQL code, correctness validations, and response times.
- **🔎 Analysis Insight**: Multi-agent collaborative discussion (powered by Microsoft AutoGen) exploring and interpreting exported charts and data.

*(The Analysis Insight page can also be run independently: `streamlit run src/analysis.py`)*

#