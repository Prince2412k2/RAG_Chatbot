from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank


class Chat:
    def __init__(self) -> None:

        embed_model = HuggingFaceEmbedding(
            model_name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        llm = Ollama(model="qwen2:0.5b", request_timeout=360.0)


        # Load Retriever
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )

        # Reranking
        rerank = SentenceTransformerRerank(model="colbert-ir/colbertv2.0", top_n=3)
        self.query_engine = index.as_query_engine(
            llm=llm, streaming=True, similarity_top_k=2, node_postprocessors=[rerank]
        )

        while True:
            query=input("ask: ")
            if query.lower()=="exit":
                break
            streaming_response = self.query_engine.query(query)
            streaming_response.print_response_stream()

model=Chat()