# agent.py
# A tiny tool-calling agent: the model decides WHEN to call a tool,
# we run the tool, feed the result back, and repeat until it has an answer.
# Run:  python agent.py   (type 'quit' to stop)

import ast
import operator
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from scraper import fetch_website_contents  # reuse the scraper from earlier

load_dotenv()


# ---------- Tools ----------
# The docstring is what the model reads to decide when to use each tool,
# so write it clearly.

# Safe math: parse the expression and allow only basic arithmetic (no eval)
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as '12 * (3 + 4)'. Supports + - * / % and **."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


@tool
def read_website(url: str) -> str:
    """Fetch the text content of a web page so it can be summarized or questioned."""
    return fetch_website_contents(url)[:6000]  # cap length to respect token limits


tools = [calculator, get_current_time, read_website]
tools_by_name = {t.name: t for t in tools}

# ---------- Model ----------
# bind_tools tells the model which tools exist and how to call them
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0).bind_tools(tools)

SYSTEM = (
    "You are a helpful assistant with tools. Use a tool when it helps "
    "(math, current time, reading a web page). Otherwise answer directly. "
    "Keep answers short."
)

MAX_STEPS = 5  # safety cap so the loop can never run forever


# ---------- The agent loop ----------
def run_agent(question: str) -> str:
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]

    for _ in range(MAX_STEPS):
        ai = model.invoke(messages)   # 1. model thinks
        messages.append(ai)

        if not ai.tool_calls:         # 2. no tool requested -> final answer
            return ai.content

        for call in ai.tool_calls:    # 3. run each requested tool
            print(f"   [tool] {call['name']}({call['args']})")
            tool_fn = tools_by_name.get(call["name"])
            result = tool_fn.invoke(call["args"]) if tool_fn else "Error: unknown tool"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        # 4. loop again so the model can use the tool results

    return "Stopped: reached the maximum number of steps."


if __name__ == "__main__":
    print("Tiny agent ready (type 'quit' to exit).")
    while True:
        q = input("You: ")
        if q.strip().lower() in {"quit", "exit"}:
            break
        if not q.strip():
            continue
        try:
            print("Agent:", run_agent(q))
        except Exception as e:
            print("Error:", e)