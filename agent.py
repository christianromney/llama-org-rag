from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from datetime import datetime
import pytz


# Agent Tools
def current_date() -> str:
  "Gets the current time as a string."
  return datetime.now(pytz.timezone("US/Eastern")).strftime("%A, %B %d, %Y")


def current_time() -> str:
  "Gets the current time as a string."
  return datetime.now(pytz.timezone("US/Eastern")).strftime("%I:%M%p %Z")


llm = Ollama(model="mistral", request_timeout=60.0)
query_engine = None  # TODO: this is not working yet

toolbox = [
  FunctionTool.from_defaults(
    fn=current_date, name="current_date", description=("Returns today's date.")
  ),
  FunctionTool.from_defaults(
    fn=current_time, name="current_time", description=("Returns the current time.")
  ),
  QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="my_notes",
    description=(
      "Returns information about my personal org-mode notes and todo list items."
      "Use a detailed plain text question as input to the tool."
    ),
  ),
]
agent = ReActAgent.from_tools(toolbox, llm=Settings.llm, verbose=True)
