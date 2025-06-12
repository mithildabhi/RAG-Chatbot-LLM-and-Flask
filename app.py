import os
from flask import Flask, request, render_template ,session,redirect, url_for
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import google.generativeai as genai
from flask_session import Session


# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Models and Vector DB setup
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = genai.GenerativeModel("gemini-1.5-flash")
client = QdrantClient(":memory:")

# Load & embed data
def load_docs():
    with open("data/marks.csv", "r", encoding="utf-8") as f:
        return f.read().split("\n\n")

documents = load_docs()
doc_embeddings = embed_model.encode(documents)

client.recreate_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

client.upload_collection(
    collection_name="docs",
    vectors=doc_embeddings,
    payload=[{"text": doc} for doc in documents],
    ids=list(range(len(documents)))
)

# Flask App
app = Flask(__name__)
app.config["SECRET_KEY"] = "mithildabhi"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []

    answer = ""
    if request.method == "POST":
        user_input = request.form["user_input"]

        # Embed and search relevant context
        query_embedding = embed_model.encode([user_input])[0]
        results = client.search(
            collection_name="docs",
            query_vector=query_embedding,
            limit=3
        )
        context = "\n".join([r.payload["text"] for r in results])

        # Add chat history as context
        history_text = "\n".join(
            [f"Q: {h['question']}\nA: {h['answer']}" for h in session["history"][-3:]]
        )

        full_prompt = f"""You are a helpful assistant. Below is the conversation history and context.\n\nConversation:\n{history_text}\n\nContext:\n{context}\n\nNew Question: {user_input}\n\nAnswer:"""

        response = llm.generate_content(full_prompt)
        answer = response.text.strip()

        # Save to session history
        session["history"].append({"question": user_input, "answer": answer})

    return render_template("index.html", answer=answer, history=session["history"])

@app.route("/clear", methods=["GET", "POST"])
def clear():
    session.pop("history", None)  # Remove if exists, ignore if not
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
