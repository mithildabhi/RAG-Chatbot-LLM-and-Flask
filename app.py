import os
from flask import Flask, request, render_template ,session,redirect, url_for
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import google.generativeai as genai
from flask_session import Session
from werkzeug.utils import secure_filename
import pandas as pd

df = pd.DataFrame()  # Initialize an empty DataFrame to avoid errors if not loaded
# Flask App Starts here
app  = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Ensure the uploads folder exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'docx', 'xlsx', 'pptx'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load API key here
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Models and Vector DB 
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = genai.GenerativeModel("gemini-1.5-flash") # generative model of gemini used for generating answers
# Initialize Qdrant client
client = QdrantClient(":memory:")

# Load and embed data
def load_docs():
    filepath = session.get("uploaded_file_path")
    
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext == ".csv" or ext == ".xlsx":
        print(" Reading CSV file...")
        df = pd.read_csv(filepath, encoding="latin1")
        # print(df.columns)
        # docs = df.head(100).to_dict(orient='records')
        # docs = df
        # Convert each row to a comma-separated string of values
        docs = df.astype(str).apply(", ".join, axis=1).tolist()
        print(docs)
    else:
        with open(filepath, "r", encoding="latin1") as f:
            docs = f.read().split("\n\n")
            
    if filepath and os.path.exists(filepath):
        print("-- Using uploaded file:", filepath)
    else:  
        filepath = "data/marks.csv"  # fallback to default
        print("-- Using default file:", filepath)

    return docs

# Flask app setup
app.config["SECRET_KEY"] = "mithildabhi"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []
    
    answer = "" # for user input answer history
    if request.method == "POST":        
        user_input = request.form["user_input"]
        if user_input:            
            if "uploaded_file_path" not in session or not session["uploaded_file_path"]:
                print(" ...No uploaded file found — using default data...")

            # Load and embed
            uploaded_data = load_docs()
            embeddings = embed_model.encode(uploaded_data)
            # Create collection
            client.recreate_collection(
                collection_name="docs",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            client.upload_collection(
                collection_name="docs",
                vectors=embeddings,
                payload=[{"text": t} for t in uploaded_data],
                ids=list(range(len(uploaded_data)))
            )
            print(" Collection uploaded to Qdrant.")

            # Embed and search relevant context
            query_embedding = embed_model.encode([user_input])[0]
            results = client.search(
                collection_name="docs",
                query_vector=query_embedding,
                limit=3
            )
            # Access and join the 'text' field from the payload dictionary in each result
            context = "\n".join([r.payload["text"] for r in results])

            # context = "\n".join([result.get("payload", {}).get("text", "") for result in results])

            # Add chat history as context
            history_text = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in session["history"][-3:]]
            )

            full_prompt = f"""You are a helpful assistant. Below is the conversation history and context.\n\nConversation:\n{history_text}\n\nContext:\n{context}\n\nNew Question: {user_input}\n\nAnswer:"""
            response = llm.generate_content(full_prompt)
            answer = response.text.strip()

            # Save to session history
            session["history"].append({"question": user_input, "answer": answer})
            
        # user input        
        return render_template("index.html", answer=answer, history=session["history"])
    
    return render_template("index.html", answer=answer, history=session["history"])    

@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "history" not in session:
            session["history"] = []

    if 'file' not in request.files:
        return redirect(request.url) # Or render an error

    file = request.files['file']
    if file.filename == '': 
        return redirect(request.url) # Or render an error

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session["uploaded_file_path"] = filepath
        
        # Load and embed
        uploaded_data = load_docs()
        embeddings = embed_model.encode(uploaded_data)
        # Create collection
        client.recreate_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        client.upload_collection(
            collection_name="docs",
            vectors=embeddings,
            payload=[{"text": t} for t in uploaded_data],
            ids=list(range(len(uploaded_data)))
        )
        print(" Collection uploaded to Qdrant.")
        return redirect(url_for("index"))
    return redirect(url_for("index"))


@app.route("/clear", methods=["GET", "POST"])
def clear():
    session.pop("history", None)  # Remove if exists, ignore if not
    session.pop("uploaded_file_path", None)  # ✅ clear uploaded path
    
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)