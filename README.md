# 🤖 RAG Customer Support Assistant

## 📌 Overview
An interactive generative AI support chatbot utilizing a Retrieval-Augmented Generation (RAG) pipeline to provide context-aware responses and manage multi-user sessions in real-time.

## 🛠️ Tech Stack
* **UI Framework:** Streamlit
* **LLM Orchestration:** LangChain
* **Vector Database:** Pinecone
* **Inference Engine:** Groq API
* **Language:** Python

## 📊 Data Processing & Capabilities
* **Context-Driven Accuracy:** Reduces AI hallucinations by grounding responses strictly in retrieved vector data.
* **Pipeline Optimization:** Features clean data ingestion and chunking of support documentation to maximize vector search performance—ensuring robust data handling suitable for advanced analytics.
* **Multi-Session Management:** Maintains isolated conversation histories for different users simultaneously to ensure data privacy.

## 🚀 How to Run Locally
1. Clone the repository: `git clone https://github.com/your-username/rag-customer-support-bot.git`
2. Create a `.env` file containing your `GROQ_API_KEY` and `PINECONE_API_KEY`.
3. Install requirements: `pip install -r requirements.txt`
4. Launch: `streamlit run app.py`
