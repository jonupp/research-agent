import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from langfuse.langchain import CallbackHandler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv("./.env")

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing from .env")

MODEL_NAME = "gemini-3.5-flash-lite"
QDRANT_URL = "http://localhost:6333/"
COLLECTION_NAME = "arxiv_abstracts"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

MAX_RAG_LOOKUPS = 4
RETRIEVAL_K = 8


# --------------------------------------------------
# Structured output schemas
# --------------------------------------------------

class RewrittenQuery(BaseModel):
    query: str = Field(
        description=(
            "A concise scientific literature search query "
            "for similarity search against a repository of abstracts."
        )
    )


class GradedDocument(BaseModel):
    arxiv_id: str = Field(description="The arXiv ID of the document.")
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Document relevance score between 0 and 1."
    )


class GradedDocuments(BaseModel):
    graded_documents: list[GradedDocument]


class RetrievalGapAnalysis(BaseModel):
    retrieval_gap_present: bool = Field(
        description=("Whether another retrieval round is necessary."))
    retrieval_gap_explanation: str = Field(description="Explanation of the retrieval gap decision.")


# --------------------------------------------------
# LLM initialization
# --------------------------------------------------

model = init_chat_model(MODEL_NAME, model_provider="google_genai")

query_suggester = model.with_structured_output(RewrittenQuery)
document_grader = model.with_structured_output(GradedDocuments)
retrieval_gap_analyzer = model.with_structured_output(RetrievalGapAnalysis)


# --------------------------------------------------
# Vector store initialization
# --------------------------------------------------

embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = QdrantVectorStore.from_existing_collection(
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    content_payload_key="document",
    metadata_payload_key="metadata",
)

retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})


# --------------------------------------------------
# LangGraph state
# --------------------------------------------------

class ResearchAssistantState(TypedDict):
    user_question: str
    number_rag_lookups: int
    queries: list[str]
    graded_docs: list[dict]
    new_docs: list[Document]
    new_graded_docs: list[dict]
    retrieval_gap_present: bool
    retrieval_gap_explanation: str
    output: list[dict]


# --------------------------------------------------
# Graph nodes
# --------------------------------------------------

def generate_query(state: ResearchAssistantState) -> dict:
    response = query_suggester.invoke([
        SystemMessage(
            content=(
                "You are a scientific literature research assistant. "
                "Rewrite the user's question into a concise and "
                "accurate semantic search query for a vector database "
                "containing scientific abstracts."
            )
        ),
        HumanMessage(content=f"User question: {state['user_question']}")
    ])

    return {"queries": state["queries"] + [response.query]}


def fetch_docs(state: ResearchAssistantState) -> dict:
    documents = retriever.invoke(state["queries"][-1])

    return {
        "new_docs": documents,
        "number_rag_lookups": state["number_rag_lookups"] + 1
    }


def grade_new_docs(state: ResearchAssistantState) -> dict:
    documents_text = "\n\n".join(
        f"arXiv ID: {doc.metadata['arxiv_id']}\n"
        f"Abstract: {doc.page_content}"
        for doc in state["new_docs"]
    )

    response = document_grader.invoke([
        SystemMessage(
            content=(
                "You are a scientific literature relevance evaluator. "
                "Evaluate each document according to its relevance "
                "to the user's question. Assign a score between "
                "0 and 1. Return the arXiv ID and relevance score "
                "for every provided document. Do not invent IDs."
            )
        ),
        HumanMessage(
            content=(
                f"User question: {state['user_question']}\n\n"
                f"Documents:\n{documents_text}"
            )
        )
    ])

    # Match LLM scores to the original Document objects.
    doc_lookup = {doc.metadata["arxiv_id"]: doc for doc in state["new_docs"]}

    graded_documents = []

    for graded_doc in response.graded_documents:
        doc = doc_lookup.get(graded_doc.arxiv_id)

        if doc is not None:
            graded_documents.append({
                "document": doc,
                "score": graded_doc.relevance_score
            })

    return {"new_graded_docs": graded_documents}


def merge_docs(state: ResearchAssistantState) -> dict:
    # Copy the existing list to avoid mutating graph state.
    all_docs = list(state["graded_docs"])

    existing_ids = {item["document"].metadata["arxiv_id"] for item in all_docs}

    for graded_doc in state["new_graded_docs"]:
        arxiv_id = graded_doc["document"].metadata["arxiv_id"]

        if arxiv_id not in existing_ids:
            all_docs.append(graded_doc)
            existing_ids.add(arxiv_id)

    return {"graded_docs": all_docs}


def analyze_gaps(state: ResearchAssistantState) -> dict:
    documents_text = "\n\n".join(
        (
            f"arXiv ID: {item['document'].metadata['arxiv_id']}\n"
            f"Score: {item['score']}\n"
            f"Abstract: {item['document'].page_content}"
        )
        for item in state["graded_docs"]
    )

    response = retrieval_gap_analyzer.invoke([
        SystemMessage(
            content=(
                "You evaluate the coverage of retrieved scientific "
                "literature. Determine whether the retrieved papers "
                "sufficiently cover the user's research question. "
                "A retrieval gap exists when important aspects of "
                "the question are insufficiently represented and "
                "another search could reasonably improve coverage."
            )
        ),
        HumanMessage(
            content=(
                f"User question: {state['user_question']}\n\n"
                f"Retrieved documents:\n{documents_text}"
            )
        )
    ])

    return {
        "retrieval_gap_present": response.retrieval_gap_present,
        "retrieval_gap_explanation": response.retrieval_gap_explanation
    }


def query_expansion(state: ResearchAssistantState) -> dict:
    previous_docs = "\n\n".join(
        (
            f"arXiv ID: {item['document'].metadata['arxiv_id']}\n"
            f"Abstract: {item['document'].page_content}"
        )
        for item in state["graded_docs"]
    )

    response = query_suggester.invoke([
        SystemMessage(
            content=(
                "Generate a new scientific literature search query. "
                "The query should explore aspects of the research "
                "question that may not have been adequately covered "
                "by previous searches. Avoid repeating previous queries."
            )
        ),
        HumanMessage(
            content=(
                f"User question: {state['user_question']}\n\n"
                f"Previous queries: {state['queries']}\n\n"
                f"Previously retrieved papers:\n{previous_docs}\n\n"
                f"Retrieval gap: {state['retrieval_gap_explanation']}"
            )
        )
    ])

    return {"queries": state["queries"] + [response.query]}


def create_output(state: ResearchAssistantState) -> dict:
    sorted_docs = sorted(
        state["graded_docs"],
        key=lambda item: item["score"],
        reverse=True
    )

    return {"output": sorted_docs}


# --------------------------------------------------
# Graph construction
# --------------------------------------------------

def routing_function(state: ResearchAssistantState) -> str:
    if state["retrieval_gap_present"] and state["number_rag_lookups"] < MAX_RAG_LOOKUPS:
        return "gap"

    return "no_gap"


workflow = StateGraph(ResearchAssistantState)

workflow.add_node("generate_query", generate_query)
workflow.add_node("fetch_docs", fetch_docs)
workflow.add_node("grade_new_docs", grade_new_docs)
workflow.add_node("merge_docs", merge_docs)
workflow.add_node("analyze_gaps", analyze_gaps)
workflow.add_node("query_expansion", query_expansion)
workflow.add_node("create_output", create_output)

workflow.set_entry_point("generate_query")

workflow.add_edge("generate_query", "fetch_docs")
workflow.add_edge("fetch_docs", "grade_new_docs")
workflow.add_edge("grade_new_docs", "merge_docs")
workflow.add_edge("merge_docs", "analyze_gaps")

workflow.add_conditional_edges(
    "analyze_gaps",
    routing_function,
    {
        "gap": "query_expansion",
        "no_gap": "create_output"
    }
)

workflow.add_edge("query_expansion", "fetch_docs")
workflow.add_edge("create_output", END)

app = workflow.compile()

# Generate graph visualization
#png_bytes = app.get_graph().draw_mermaid_png()
#with open("graph_visualization.png", "wb") as f:
#    f.write(png_bytes)
#print("Graph saved successfully as 'graph_visualization.png'!")

# --------------------------------------------------
# Execution
# --------------------------------------------------
user_question = input("Enter your research question: ")

initial_state: ResearchAssistantState = {
    "user_question": (
        {user_question}
    ),
    "number_rag_lookups": 0,
    "queries": [],
    "graded_docs": [],
    "new_docs": [],
    "new_graded_docs": [],
    "retrieval_gap_present": False,
    "retrieval_gap_explanation": "",
    "output": []
}

langfuse_handler = CallbackHandler()
result = app.invoke(initial_state, config={"callbacks": [langfuse_handler]})

print("-" * 80)
print(f"Number of RAG fetches: {result["number_rag_lookups"]}")
print("-" * 80)
print(f"Queries: {result["queries"]}")
print("-" * 80)

for item in result["output"]:
    doc = item["document"]

    print(f"arXiv ID: {doc.metadata['arxiv_id']}")
    print(f"Relevance score: {item['score']:.3f}")
    print(f"Abstract: {doc.page_content}")
    print("-" * 80)
