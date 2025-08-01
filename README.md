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
   
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
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

<img width="1713" height="726" alt="Screenshot 2025-07-12 195025" src="https://github.com/user-attachments/assets/59e32464-f216-4d94-b4dc-091513247b2f" />

<img width="1693" height="573" alt="Screenshot 2025-07-12 195106" src="https://github.com/user-attachments/assets/ceef8702-794f-4f22-9386-0274e4f0022b" />

