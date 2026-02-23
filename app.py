import os
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config

load_dotenv()

app_flask = Flask(__name__)

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


embeddings = get_embeddings_model()


def get_vector_store(embed_func):
    return Chroma(
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embed_func
    )


vectordb = get_vector_store(embeddings)


def get_chat_model():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=400
    )


model = get_chat_model()


def call_model(state: MessagesState):
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


def get_langgraph_app():
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()
    app_graph = workflow.compile(checkpointer=memory)
    return app_graph


graph_app = get_langgraph_app()

thread_id = "flask_chat_session"


@app_flask.route("/")
def home():
    return render_template("index.html")


@app_flask.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    try:
        docs = vectordb.similarity_search_with_score(user_input, k=3)

        _docs = pd.DataFrame(
            [
                (
                    user_input,
                    doc[0].page_content,
                    doc[0].metadata.get("source"),
                    doc[0].metadata.get("page"),
                    doc[1],
                )
                for doc in docs
            ],
            columns=["query", "paragraph", "document", "page_number", "score"],
        )

        current_context = "\n\n".join(_docs["paragraph"])

        current_turn_message = HumanMessage(
            content=f"Context: {current_context}\n\nQuestion: {user_input}"
        )

        result = graph_app.invoke(
            {"messages": [current_turn_message]},
            config={"configurable": {"thread_id": thread_id}},
        )

        ai_response = result["messages"][-1].content

        source_document = (
            _docs["document"][0] if not _docs.empty else "N/A"
        )

        top_pages = (
            _docs["page_number"]
            .drop_duplicates()
            .head(3)
            .astype(str)
            .tolist()
        )

        page_numbers_str = ", ".join(top_pages) if top_pages else "N/A"

        final_response = (
            f"{ai_response}\n\nSource Document: {source_document}\n"
            f"Reference Pages: {page_numbers_str}"
        )

        return jsonify({"response": final_response})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})


if __name__ == "__main__":
    app_flask.run(debug=True)