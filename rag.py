from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import Document, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
import os, getopt, sys

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=30.0)

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
      self.index = VectorStoreIndex.from_documents(self.documents,
                                                   show_progress=progress)
      self.save()

  def refresh(self):
    "Refreshes the index from the updated documents and saves to disk."
    log("Refreshing index with changed documents")
    self.index.refresh(self.documents)
    self.save()

  def save(self):
    "Saves the index to disk under the given directory."
    log("Saving index to disk")
    self.index.storage_context.persist(persist_dir=self.path)

  def print_files(self):
    "Prints the list of all files in the index."
    files = [info.metadata["file_path"] for info in self.index.ref_doc_info.values()]
    print("\n".join(files))

  def query(self, q):
    "Returns the response to the given query."
    return self.index.as_query_engine().query(q)

  def chat(self, mode="context", stream=True):
    engine = self.index.as_chat_engine(chat_mode=mode, streaming=stream)
    engine.streaming_chat_repl()
    return engine

if __name__ == "__main__":
  # default values
  verbose = False
  interactive = False
  refresh = False
  listing = False
  query = ''
  directory = "/Users/christian/Documents/personal/notes/content/"

  arguments = sys.argv[1:]
  short_opts = 'virlq:d:'
  long_opts = ['verbose', 'interactive', 'refresh', 'list', 'query=', 'directory=']

  try:
    opts, _args = getopt.getopt(arguments, short_opts, long_opts)
    for opt, arg in opts:
      if opt in ('-v', '--verbose'):
        verbose = True
      elif opt in ('-i', '--interactive'):
        interactive = True
      elif opt in ('-r', '--refresh'):
        refresh = True
      elif opt in ('-l', '--list'):
        listing = True
      elif opt in ('-q', '--query'):
        query = arg
      elif opt in ('-d', '--directory'):
        directory = arg

    index = DocumentIndex(directory)
    if listing:
      index.print_files()
    elif interactive:
      index.chat()
    elif query:
      print(index.query(query))

    if refresh:
      index.refresh()

    log("Goodbye.")
  except getopt.GetoptError as err:
    print(str(err))
    sys.exit(2)
