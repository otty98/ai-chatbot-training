import { useState } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleSendMessage = async (messageToSend = input) => {
    if (messageToSend.trim() === '') return;

    // Add the user's message to the chat
    const userMessage = { text: messageToSend, sender: 'user' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');

    // Connect to your Flask backend
    try {
      const response = await fetch('http://127.0.0.1:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          session_id: 'test_session_123',
          user_id: 'test_user_456',
        }),
      });

      const data = await response.json();
      const botMessage = { ...data, sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, botMessage]);

    } catch (error) {
      console.error("Error connecting to the backend:", error);
      const errorMessage = { text: "Sorry, I couldn't connect to the server.", sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }
  };

  const handleFollowUpClick = (question) => {
    handleSendMessage(question);
  };

  return (
    <div className="chat-container">
      <div className="messages-box">
        {messages.map((msg, index) => {
          // Check if it's a user message, as it won't have the extra fields
          if (msg.sender === 'user') {
            return (
              <div key={index} className={`message ${msg.sender}`}>
                <p>{msg.text}</p>
              </div>
            );
          }
          
          // Render bot messages with all the extra fields
          return (
            <div key={index} className={`message ${msg.sender}`}>
              <p>{msg.response}</p>
              {msg.multimedia && (
                <div className="multimedia">
                  <img src={msg.multimedia.url} alt={`Multimedia: ${msg.multimedia.type}`} />
                </div>
              )}
              {msg.related_concepts && msg.related_concepts.length > 0 && (
                <div className="related-concepts">
                  <strong>Related Concepts:</strong>
                  <ul>
                    {msg.related_concepts.map((concept, i) => (
                      <li key={i}>{concept}</li>
                    ))}
                  </ul>
                </div>
              )}
              {msg.follow_ups && msg.follow_ups.length > 0 && (
                <div className="follow-ups">
                  <strong>Follow-up Questions:</strong>
                  <ul>
                    {msg.follow_ups.map((follow_up, i) => (
                      <li key={i} onClick={() => handleFollowUpClick(follow_up)}>
                        {follow_up}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="input-box">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              handleSendMessage();
            }
          }}
          placeholder="Type a message..."
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;