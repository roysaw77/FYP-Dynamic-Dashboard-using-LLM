# Dynamic Dashboard using LLM

## Overview
A Final Year Project (FYP) dynamic dashboard built around **Text-to-SQL over uploaded CSV files** with multi-agent interpretation.

- **Live Web App**: [https://fyp-dynamic-dashboard-using-llm.streamlit.app/](https://fyp-dynamic-dashboard-using-llm.streamlit.app/)
- **Research Paper**: [thesis (1).pdf](https://github.com/user-attachments/files/27086468/thesis.1.pdf)

---

## Demo Video

https://github.com/user-attachments/assets/669048f9-6b04-48d3-ad20-7755b33b3314



---

## Key Features
- **LLM-driven dashboard generation**: Automatically generate SQL queries and data visualizations based on natural language input.
- **Automated Validation**: Real-time evaluation of generated SQL and chart responses using secondary LLM judgment.
- **Multi-Agent Insight Analysis**: Collaborative discussion among specialized agents (ChartAnalyst, DataInterpreter, Critic, Summarizer) powered by AutoGen.
- **Customizable LLM Backend**: Powered by NVIDIA NIM with models including Nemotron-3.5-30B, Llama-3.1-405B, Llama-3.1-8B, and Mistral-Large.

---

## System Architecture & Framework
- **Frontend**: Streamlit
- **Text-to-SQL Engine**: PandasAI 3.0 + LiteLLM + NVIDIA NIM
- **Multi-Agent Analysis**: Microsoft AutoGen 0.2.x + Vision LLM

### Dynamic Dashboard Generation Pipeline
<img width="818" height="420" alt="Dashboard Generation Pipeline" src="https://github.com/user-attachments/assets/2ff4dc74-45d6-4be5-ad2a-b03845c5255b" />

### Multi-Agent Analysis Framework
![Multi-Agent Analysis Framework](docs/assets/Multi-Agent%20Analysis%20Framework.png)

---

## Project Structure
```
FYP-Dynamic-Dashboard-using-LLM/
├── src/                    # Streamlit application source files
│   ├── main.py             # Main entry point and sidebar router
│   ├── dashboard.py        # PandasAI generation, log scraping, validation
│   └── analysis.py         # AutoGen multi-agent chart analysis panel
├── docs/                   # Documentation and media assets
│   └── assets/             # Demo videos, diagrams, and UI screenshots
├── requirements.txt        # Pinned Python dependencies
└── README.md
```

> **Note**: Benchmark data (`dataset/`, `exports/`, and `test_model/`) are archived in the `backup-full` branch and ignored on `main` to maintain a lightweight, clean repository.

---

## Setup Instructions

### Prerequisites
- **Python 3.11** (Required: `>=3.10, <3.12` for `pandasai-litellm`; Python 3.11.9 recommended).
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

---

## How to Run in Streamlit

> **Important**: Always launch commands from the **repository root** (`FYP-Dynamic-Dashboard-using-LLM`) so that relative paths and chart exports resolve correctly.

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

---
