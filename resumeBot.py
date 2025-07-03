from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import tempfile

app = Flask(__name__)
CORS(app)  # ✅ Allow cross-origin requests from frontend

@app.route('/ask', methods=['POST'])
def ask():
    if 'resume' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'Missing file or question'}), 400

    resume_file = request.files['resume']
    question = request.form['question']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        resume_file.save(tmp.name)
        reader = PDFReader()
        documents = reader.load_data(file=tmp.name)

    # Set embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build index and query
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    return jsonify({'answer': str(response)})

if __name__ == '__main__':
    print("✅ ResumeBot is running. Visit http://127.0.0.1:5000")
    app.run(debug=True)
