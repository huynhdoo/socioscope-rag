from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from pydantic import BaseModel, Field

system_prompt = (
    "You're a helpful AI assistant."
    "Given a user question and some documents, answer the user question."
    "If none of the documents answer the question, just say you don't know."
    "\n\nHere are the project documents: "
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# Cite snippets
class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )

class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )

class State(TypedDict):
    question: str
    context: List[Document]
    answer: QuotedAnswer

def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\Project name: {doc.metadata['NAME']}\nInterview Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)

def create_llm_graph(db, api_key):
    llm = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=api_key)
    retriever = db.as_retriever(search_kwargs={"k": 100})

    def retrieve(state: State):
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        formatted_docs = format_docs_with_id(state["context"])
        messages = prompt.invoke({"question": state["question"], "context": formatted_docs})
        structured_llm = llm.with_structured_output(QuotedAnswer)
        response = structured_llm.invoke(messages)
        return {"answer": response}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph