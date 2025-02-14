# Open-Deep-Research--Smolagents 🚀🔍

A brief description of your project and what it does. This project leverages multiple AI models and tools to answer questions by running an agent that queries external sources.

## Overview 📖

This project integrates multiple AI models and tools, including both reasoning and code agents, to perform deep research on queries. With its CLI interface, you can interact with the agent directly from your terminal.

## Features ✨

- **Command Line Interface (CLI):** Easily run the agent and input your query directly from the terminal.
- **Multiple AI Models Integration:** Utilizes both reasoning and code agents for robust query processing.
- **Tool Support:** Integrates various web and text inspection tools to gather and verify information.
- **Extensible:** Easily add new tools or modify agent hierarchies as needed.

## Installation 🛠️

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Open-Deep-Research--Smolagents.git
   cd Open-Deep-Research--Smolagents
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.7+ installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables:**

   Create a `.env` file in the project root with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   HF_TOKEN=your_huggingface_token
   SERPAPI_API_KEY=your_serpapi_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

   Replace the placeholder values with your actual API keys.

## Usage 🚀

Run the project from the command line:

```bash
  python run.py "Your query goes here"
```

If you do not provide a query as an argument, the script will prompt you to enter one interactively:

```bash
  python run.py
```

Then, enter your query when prompted.

## Project Structure 🗂️

- `run.py`: The main entry point for running the project.
- `requirements.txt`: Contains all Python dependencies.
- `scripts/`: Contains various modules and tools used by the agent.
- `.env`: File to store your API keys and configuration (do not commit this file).

## Contributing 🤝

Contributions are welcome! Please open an issue or submit a pull request for improvements.
