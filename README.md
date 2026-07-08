# AI Education Chatbot

A full-stack chatbot that helps people learn about AI. It explains concepts clearly, points out related topics worth exploring, and adapts to what each user already knows.

## 🤖 Features

- **Context-aware answers** — combines a knowledge base with a generative AI model to give relevant, helpful responses
- **Guided learning paths** — structured flows for topics like AI fundamentals, NLP, and AI ethics
- **Progress tracking** — saves conversation history and tailors recommendations based on where each user is at
- **Clean UI** — built with React, responsive and easy to use

## 💻 Tech stack

**Frontend:** React, Vite, CSS

**Backend:** Flask, Python, Flask-CORS, `google-generativeai`, `pymongo`

**Database:** MongoDB — stores conversations, the knowledge base, and user progress

## Getting started

### 1. Backend setup

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in the backend directory with your MongoDB connection string and Gemini API key:

```
MONGODB_URI='mongodb://localhost:27017/'
GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
```

Start the server:

```bash
python app.py
```

It'll run at `http://127.0.0.1:5000`.

### 2. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173` and will connect to the backend automatically.
