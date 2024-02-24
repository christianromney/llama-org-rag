from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import FixedRecencyPostprocessor
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
import os, getopt, sys, shutil

# global settings
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=60.0)

verbose=False
def log(message):
  "Prints a message iff verbose is True"
  if verbose:
    print(message)

class DocumentIndex:
  def __init__(self, directory, exts=[".org"], progress=True,
               verbose=False, max_top_k=10, top_k=5,
               similarity_cutoff=0.6):
    # load or create the index
    self.path = os.path.join(directory, ".llamaindex")
    if os.path.exists(self.path):
      context= StorageContext.from_defaults(persist_dir=self.path)
      log("Loading index from disk")
      self.index = load_index_from_storage(context)
    else:
      log("Creating new index")
      docs = SimpleDirectoryReader(input_dir=directory,
                                   recursive=True,
                                   required_exts=exts).load_data()
      log(f"Read {len(docs)} {', '.join(exts)} docs from {directory}.")
      index = VectorStoreIndex.from_documents(docs, show_progress=progress)
      index.storage_context.persist(persist_dir=self.path)
      self.index = index

    # post-processors filter the nodes returned from the similarity search
    # prior to creating the context for the LLM call
    self.post_processors = [
      SimilarityPostprocessor(similarity_cutoff=similarity_cutoff),
      FixedRecencyPostprocessor(top_k=top_k, date_key='last_modified_date')
    ]

    # compact and refine synthesizer
    self.response_synth = get_response_synthesizer()

    self.query_engine = CitationQueryEngine(
      retriever=VectorIndexRetriever(
        index=self.index, similarity_top_k=top_k,
        max_top_k=max_top_k),
      response_synthesizer=self.response_synth,
      node_postprocessors=self.post_processors
    )
    self.chat_engine = CondenseQuestionChatEngine.from_defaults(
      query_engine=self.query_engine
    )

  def delete_index(self):
    "Deletes the index folder and all files"
    shutil.rmtree(self.path)

  def print_files(self):
    "Prints the list of all files in the index."
    files = [info.metadata["file_path"] for info in self.index.ref_doc_info.values()]
    print("\n".join(files))

  def _print_retrieved_item(self, idx, item):
    print(f"{idx}. file: {item.node.metadata['file_path']}")
    print(f"   size: {item.node.metadata['file_size']}")
    print(f"   score: {round(item.score, 3)}")
    print(f"   embedding: {item.node.embedding}")

  def _print_retrieved_items(self, items):
    heading = f"\nRetrieved {len(items)} Nodes"
    dashes = "-" * len(heading)
    print(f"{heading}\n{dashes}")
    for i in range(len(items)):
      self._print_retrieved_item(i + 1, items[i])

  def print_retrieved(self, q):
    "Print nodes retrieved from the index."
    items = self.query_engine.retriever.retrieve(q)
    self._print_retrieved_items(items)

  def query(self, q, evaluate_result=False):
    "Prints the response to the given query."
    print(self.query_engine.query(q))

  def chat(self, mode="context", stream=True):
    "Starts a chat repl."
    self.chat_engine.streaming_chat_repl()

if __name__ == "__main__":
  # default values
  interactive = False
  print_retrieved = False
  listing = False
  evaluate = False
  query = ''
  directory = "/Users/christian/Documents/personal/notes/content/"

  # argument parsing
  arguments = sys.argv[1:]
  short_opts = 'vipleq:d:'
  long_opts = ['verbose', 'interactive', 'print-retrieved',
               'list', 'eval', 'query=', 'directory=']

  try:
    opts, _args = getopt.getopt(arguments, short_opts, long_opts)
    for opt, arg in opts:
      if opt in ('-v', '--verbose'):
        verbose = True
      elif opt in ('-i', '--interactive'):
        interactive = True
      elif opt in ('-p', '--print-retrieved'):
        print_retrieved = True
      elif opt in ('-l', '--list'):
        listing = True
      elif opt in ('-e', '--eval'):
        evaluate = True
      elif opt in ('-q', '--query'):
        query = arg
      elif opt in ('-d', '--directory'):
        directory = arg

    # RAG class
    index = DocumentIndex(directory)

    # dispatch action
    if listing:
      index.print_files()
    elif interactive:
      agent.chat()
    elif query:
      index.query(query, evaluate_result=evaluate)
      if print_retrieved:
        index.print_retrieved(query)
  except getopt.GetoptError as err:
    print(str(err))
    sys.exit(2)
