from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
import os

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=30.0)

def acknowledged(question):
  "Returns true if the answer to the question is 'y'."
  answer = input(question +  "(y|n)? " )
  return answer.lower().strip()[0] == "y"

def log(message):
  "Prints a message iff verbose is True"
  if verbose:
    print(message)

class DocumentIndex:
  def __init__(self, directory, progress=True, verbose=False):
    self.path = os.path.join(directory, ".llamaindex")
    self.documents = SimpleDirectoryReader(input_dir=directory,
                                           recursive=True,
                                           required_exts=[".org"]).load_data()
    if os.path.exists(self.path):
      context= StorageContext.from_defaults(persist_dir=self.path)
      log("Loading index from disk")
      self.index = load_index_from_storage(context)
    else:
      log("Creating new index")
      self.index = VectorStoreIndex.from_documents(documents, show_progress=progress)
      self.save()

  def refresh(self):
    log("Refreshing index with changed documents")
    self.index.refresh(self.documents)
    self.save()

  def save(self):
    log("Saving index to disk")
    self.index.storage_context.persist(persist_dir=self.path)

  def chat(self, mode="context", stream=True):
    engine = self.index.as_chat_engine(chat_mode=mode, streaming=stream)
    engine.streaming_chat_repl()
    return engine

if __name__ == "__main__":
  verbose = False
  index = DocumentIndex("/Users/christian/Documents/personal/notes/content/")
  engine = index.chat()

  if acknowledged("Refresh index"):
    index.refresh()

  log("Goodbye.")
