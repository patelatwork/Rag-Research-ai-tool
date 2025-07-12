Research Paper Q&A Tool

A Streamlit-based application that uses RAG (Retrieval-Augmented Generation) to enable interactive conversations with research papers.

 🌟 Features

- 📄 PDF Research Paper Upload
- 💡 Intelligent Question Answering
- 🔍 Context-Aware Responses
- 💬 Chat-Style Interface
- 🔄 Session History Persistence



 🚀 Getting Started

 Prerequisites

- Python 3.8+
- OpenAI API Key

 Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd RAG\ application
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

 Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser and navigate to the provided local URL (typically `http://localhost:8501`)

## 📖 How to Use

1. Upload Paper
   - Use the file uploader to select your research paper in PDF format

2. Ask Questions
   - Type your question in the input box at the bottom
   - Click "Ask" or press Enter to submit




 Document Processing
- Chunk size: 1200 characters
- Chunk overlap: 150 characters
- Custom separators for optimal text splitting
 
  Vector Search
- FAISS for efficient similarity search
- Top-5 most relevant chunks retrieved per query

 Language Model
- Uses GPT-4 for response generation
- Temperature: 1.5 for creative yet focused responses

 📝 Notes

- The application maintains chat history during the session
- Large PDF files may take a few moments to process
- Responses are generated based on the content of the uploaded paper only

 🤝 Contributing

Feel free to:
- Open issues
- Submit Pull requests
- Suggest improvements


