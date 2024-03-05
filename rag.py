from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import FixedRecencyPostprocessor
from llama_index.core.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import UnstructuredReader
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
import os, getopt, sys, logging

# global settings
logging.basicConfig(stream=sys.stdout, level=logging.WARN)
log = logging.getLogger()

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=60.0, temperature=0.5)

class DocumentIndex:
  def __init__(self, directory, exts=[".org"], progress=True,
               verbose=False, max_top_k=20, top_k=10,
               similarity_cutoff=0.6):
    self.collection = "llama-org-rag"

    # create the Qdrant client
    client = QdrantClient("localhost", port=6333)
    vector_store = QdrantVectorStore(client=client,
                                     collection_name=self.collection)

    # load or create the index?
    info = [c for c in client.get_collections().collections if c.name == self.collection]
    if 0 < len(info):
      log.info("Loading index from disk")
      self.index = VectorStoreIndex.from_vector_store(vector_store)
    else:
      log.info("Creating new index")
      docs = SimpleDirectoryReader(input_dir=directory,
                                   recursive=True,
                                   required_exts=exts).load_data()
      log.info(f"Read {len(docs)} {', '.join(exts)} docs from {directory}.")
      context = StorageContext.from_defaults(vector_store=vector_store)
      self.index = VectorStoreIndex.from_documents(docs, show_progress=progress,
                                                   storage_context=context)

    # post-processors filter the nodes returned from the similarity search
    # prior to creating the context for the LLM call
    self.post_processors = [
      SimilarityPostprocessor(similarity_cutoff=similarity_cutoff),
      FixedRecencyPostprocessor(top_k=top_k, date_key='last_modified_date')
    ]

    # query engine
    template = (
      "The following context draws from notes and other documents I have written:\n"
      "---------------------\n"
      "{context_str}"
      "\n---------------------\n"
      "Given that context, please answer this question: '{query_str}'\n\n"
      "Always cite which of my documents you used to answer my question.\n"
      "After your answer, produce an ordered list of those Citations\n"
      "containing the file_path of the corresponding document.\n"
    )
    qa_template = PromptTemplate(template)

    self.query_engine = self.index.as_query_engine(
      similarity_top_k=top_k,
      max_top_k=max_top_k,
      text_qa_template=qa_template,
      node_postprocessors=self.post_processors
    )

    self.chat_engine = CondenseQuestionChatEngine.from_defaults(
      query_engine=self.query_engine
    )

  def print_files(self):
    "Prints the list of all files in the index."
    files = [info.metadata["file_path"] for info in self.index.ref_doc_info.values()]
    print("\n".join(files))

  def query(self, q):
    "Prints the response to the given query."
    print(self.query_engine.query(q))

  def chat(self, mode="context", stream=True):
    "Starts a chat repl."
    self.chat_engine.streaming_chat_repl()

if __name__ == "__main__":
  # default values
  interactive = False
  listing = False
  query = ''
  directory = "/Users/christian/Documents/personal/notes/content/"

  # argument parsing
  arguments = sys.argv[1:]
  short_opts = 'vilq:d:'
  long_opts = ['verbose', 'interactive', 'list',
               'query=', 'directory=']

  try:
    opts, _args = getopt.getopt(arguments, short_opts, long_opts)
    for opt, arg in opts:
      if opt in ('-v', '--verbose'):
        log.setLevel(logging.DEBUG)
      elif opt in ('-i', '--interactive'):
        interactive = True
      elif opt in ('-l', '--list'):
        listing = True
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
      index.query(query)

  except getopt.GetoptError as err:
    print(str(err))
    sys.exit(2)
  except ResponseHandlingException as err:
    print("Unable to connect to qdrant. Is the qdrant container running?")
    sys.exit(4)
