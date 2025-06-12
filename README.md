
# 🧠 RAG Chatbot using Gemini, Flask & Your Data

This project is a Retrieval-Augmented Generation (RAG) chatbot using:

- **Flask** for the web interface
- **Google Gemini 1.5 Flash** for answering questions
- **SentenceTransformers** for vector embeddings
- **Qdrant (in-memory)** as the vector store
- Your **custom data (CSV, TXT)** as knowledge base

---

## 🚀 Features

- Conversational chatbot UI
- Retrieves relevant data from your documents
- Uses Gemini to generate context-aware responses
- Keeps session chat history
- Simple UI using Flask + HTML + CSS

---

## 🔧 How It Works

1. User submits a question in the UI
2. `sentence-transformers` embeds the question
3. Top 3 most relevant documents are retrieved from Qdrant
4. A prompt is built using:
    - Top retrieved docs
    - Last 3 Q&A history entries
5. Prompt is sent to Gemini LLM
6. Gemini returns an answer
7. Answer and question are stored in session

---

## 🧪 Example Output

**Q:** What is the average mark in the file?  
**A:** Based on the data, the average mark is approximately 82.3.

---

## 🗂️ File Structure

```
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── .env                    # Gemini API key (not committed)
├── data/
│   ├── marks.csv           # CSV data used for retrieval
│   └── wizard_of_oz.txt    # Example long-form text
├── static/
│   └── style.css           # CSS styling
├── templates/
│   └── index.html          # Chat UI page
```

---

## ⚙️ Setup Instructions

1. Clone the repository

```
git clone https://github.com/mithildabhi/RAG-Chatbot-LLM-and-Flask.git
cd RAG-Chatbot-LLM-and-Flask
```

2. Install requirements

```
pip install -r requirements.txt
```

3. Set up your `.env` file

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

Get your API key at https://makersuite.google.com/app/apikey

4. Run the app

```
python app.py
```

5. Open in browser:  
http://127.0.0.1:5000

---

## 📁 Customizing Data

Edit `data/marks.csv` or add `.txt` files for more context.

- Data is split by paragraphs or double newlines
- Re-run app after editing the files

---

## 🧠 Technologies Used

- Flask
- Jinja2 Templates
- Google Generative AI (Gemini)
- Sentence Transformers
- Qdrant Vector Search
- dotenv
- flask_session

---

## 📝 License

MIT License

---

## 👨‍💻 Author

Built with ❤️ by Mithil using Gemini & OpenAI tools
