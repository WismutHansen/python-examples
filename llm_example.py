#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai",
#     "python-dotenv",
#     "rich",
# ]
# ///
"""
Script with a polished TUI for LLM interaction.

Requires Python 3.11+
Dependencies:
    - openai
    - python-dotenv
    - rich

Usage:
    python3 script.py [--verbose] [--system "Your system prompt here"]
"""

import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Load environment variables
load_dotenv()
openai_base_url = os.getenv("OPENAI_BASE_URL", "http://10.161.141.2:1234/v1")
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")

client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)


def ask_LLM(messages):
    """
    Send the conversation history to the LLM using streaming and
    return the assistant's reply.

    :param messages: List of dicts, each with keys "role" and "content"
    """
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
    )
    assistant_message = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            assistant_message += chunk.choices[0].delta.content
    return assistant_message


def main():
    # Setup CLI arguments
    parser = argparse.ArgumentParser(
        description="LLM TUI Chat with context display and verbose mode."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display the full conversation context (history) for debugging.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant",
        help="Custom system prompt for the LLM.",
    )
    args = parser.parse_args()

    console = Console()
    # Initialize conversation history with the system message
    messages = [{"role": "system", "content": args.system}]

    console.print(Panel("Welcome to the LLM TUI Chat", style="bold green"))

    while True:
        try:
            # Get user input with rich prompt
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            # Append the user message to the conversation history
            messages.append({"role": "user", "content": user_input})

            # If verbose, display the entire context window
            if args.verbose:
                context_str = "\n".join(
                    f"[bold]{msg['role'].capitalize()}:[/bold] {msg['content']}"
                    for msg in messages
                )
                console.print(
                    Panel(context_str, title="Context Window", style="magenta")
                )

            # Get the LLM response based on the full conversation history
            answer = ask_LLM(messages)
            # Append the assistant's answer to the history
            messages.append({"role": "assistant", "content": answer})

            # Display the assistant's response in a panel
            console.print(Panel(answer, title="[bold green]Assistant[/bold green]"))

        except KeyboardInterrupt:
            console.print("\nExiting... Goodbye!", style="bold red")
            break
        except Exception as e:
            console.print(f"An error occurred: {e}", style="bold red")
            break


if __name__ == "__main__":
    main()
