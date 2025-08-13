AI Education Chatbot
This project is a full-stack, AI-powered educational chatbot designed to help users learn about Artificial Intelligence and related topics. The chatbot provides clear explanations, suggests related concepts, and offers personalized learning paths.

🤖 Features
Context-Aware Responses: The chatbot uses a knowledge base and a Generative AI model to provide relevant and helpful answers to user queries.

Structured Conversation Flows: Users can follow predefined learning paths on topics like AI fundamentals, NLP, and AI ethics.

User Progress Tracking: The system can save conversation history and track user progress to offer personalized recommendations.

Modern UI: A clean and responsive user interface built with React.

💻 Technologies
Frontend:

React: The main JavaScript library for building the user interface.

Vite: A fast build tool for the frontend development server.

CSS: For styling the application.

Backend:

Flask: A lightweight Python web framework to handle API requests.

Python: The core programming language for the backend logic.

Flask-CORS: Manages Cross-Origin Resource Sharing for communication between the frontend and backend.

google-generativeai: The library used to interact with the Generative AI model.

pymongo: The Python driver for MongoDB, used for database interactions.

MongoDB: The NoSQL database used to store conversations, a knowledge base, and user progress

Bash

cd backend
Install the required Python packages using pip.

Bash

pip install -r requirements.txt
Set up your environment variables. Create a .env file and add your MongoDB connection string and Gemini API key.

Bash

MONGODB_URI='mongodb://localhost:27017/'
GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
Run the Flask development server.

Bash

python app.py
The server will start on http://127.0.0.1:5000.

2. Frontend Setup
Navigate to the frontend directory.

Bash

cd frontend
Install the Node.js dependencies.

Bash

npm install
Start the Vite development server.

Bash

npm run dev
The frontend will be available at http://localhost:5173. The application will automatically connect to the backend running on port 5000.
