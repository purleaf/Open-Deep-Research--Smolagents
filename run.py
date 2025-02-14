import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    MANAGED_AGENT_PROMPT,
    CodeAgent,
    ManagedAgent,
    # HfApiModel,
    LiteLLMModel,
    OpenAIServerModel,
    Model,
    ToolCallingAgent,
    LiteLLMModelDeepSeek
)


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
def create_agent_hierarchy(tiny_model: Model, reasoning_model: Model):
    text_limit = 100000
    ti_tool = TextInspectorTool(reasoning_model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(reasoning_model, text_limit),
    ]

    text_webbrowser_agent = ManagedAgent(
        agent=ToolCallingAgent(
            model=reasoning_model,
            tools=WEB_TOOLS,
            max_steps=20,
            verbosity_level=2,
            planning_interval=4,
        ),
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
        Ask him for all your questions that require browsing the web.
        Provide him as much context as possible, in particular if you need to search on a specific timeframe!
        And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
        Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
        """,
        provide_run_summary=True,
        managed_agent_prompt=MANAGED_AGENT_PROMPT + """You can navigate to .txt online files.
        If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
        Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""
    )


    manager_agent = CodeAgent(
        model=tiny_model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent
def answer_single_question(query):
    tiny_model = OpenAIServerModel(model_id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    reasoning_model = LiteLLMModelDeepSeek(model_id="ollama/deepseek-r1:32b", api_base="http://localhost:11434")
    #model = LiteLLMModel(model_id="ollama/deepseek-r1:7b", api_base="http://localhost:11434")
    # model = LiteLLMModel(
    #     custom_role_conversions=custom_role_conversions,
    #     max_completion_tokens=8192,
    #     reasoning_effort="high",
    # )
    # model = HfApiModel("Qwen/Qwen2.5-72B-Instruct", provider="together")
    #     "https://lnxyuvj02bpe6mam.us-east-1.aws.endpoints.huggingface.cloud",
    #     custom_role_conversions=custom_role_conversions,
    #     # provider="sambanova",
    #     max_tokens=8096,
    # )
    # document_inspection_tool = TextInspectorTool(reasoning_model, 100000)

    agent = create_agent_hierarchy(tiny_model=tiny_model, reasoning_model=tiny_model)

    augmented_question = f"""You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
{query}
"""
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent 🚀
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages(summary_mode=True)

        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=tiny_model)

        output = str(final_result)
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = [str(step) for step in agent.memory.steps]

        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False

        # check if iteration limit exceeded
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_output = {
        "agent_name_reasoning": reasoning_model.model_id,
        "agent_name_code": tiny_model.model_id,
        "question": query,
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
    }
    return final_output

def main():
    parser = argparse.ArgumentParser(description="Run the agent to answer a query via CLI.")
    parser.add_argument(
        "query",
        nargs="?",
        help="The query you want to ask. If not provided, the script will prompt for input.",
    )
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        query = input("Enter your query: ")

    result = answer_single_question(query)
    # Pretty-print the JSON output
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()