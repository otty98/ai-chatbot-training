from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import google.generativeai as genai
import os
from bson.objectid import ObjectId
import json

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MONGODB_URI'] = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
app.config['DATABASE_NAME'] = 'chatbot'
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# MongoDB connection
client = MongoClient(app.config['MONGODB_URI'])
db = client[app.config['DATABASE_NAME']]

# Collections
conversations = db.conversations
knowledge_base = db.knowledge_base
user_progress = db.user_progress

class AIEducationChatbot:
    def __init__(self):
        self.knowledge_base = self.load_knowledge_base()
        self.conversation_flows = self.define_conversation_flows()
    
    def load_knowledge_base(self):
        """Load the knowledge base with AI concepts"""
        kb_data = [
            {
                "topic": "ai_fundamentals",
                "question": "What is Artificial Intelligence?",
                "answer": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
                "related_concepts": ["machine_learning", "neural_networks", "deep_learning"],
                "difficulty": "beginner",
                "multimedia": {
                    "type": "diagram",
                    "url": "/static/diagrams/ai_overview.svg"
                }
            },
            {
                "topic": "machine_learning",
                "question": "What is Machine Learning?",
                "answer": "Machine Learning (ML) is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data to make predictions or decisions.",
                "related_concepts": ["supervised_learning", "unsupervised_learning", "reinforcement_learning"],
                "difficulty": "beginner",
                "multimedia": {
                    "type": "flowchart",
                    "url": "/static/diagrams/ml_process.svg"
                }
            },
            {
                "topic": "ai_vs_ml_vs_dl",
                "question": "What's the difference between AI, ML, and Deep Learning?",
                "answer": "AI is the broadest term - it's any technique that enables machines to mimic human intelligence. ML is a subset of AI that learns from data. Deep Learning is a subset of ML that uses neural networks with multiple layers. Think of them as nested circles: AI contains ML, which contains Deep Learning.",
                "related_concepts": ["neural_networks", "algorithms"],
                "difficulty": "beginner",
                "multimedia": {
                    "type": "comparison_chart",
                    "url": "/static/diagrams/ai_ml_dl_comparison.svg"
                }
            },
            {
                "topic": "neural_networks",
                "question": "What are Neural Networks?",
                "answer": "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information by passing signals through weighted connections. They're the foundation of deep learning.",
                "related_concepts": ["deep_learning", "perceptrons", "backpropagation"],
                "difficulty": "intermediate",
                "multimedia": {
                    "type": "interactive_diagram",
                    "url": "/static/diagrams/neural_network.svg"
                }
            },
            {
                "topic": "nlp",
                "question": "What is Natural Language Processing?",
                "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language. NLP combines computational linguistics with statistical and machine learning models to enable machines to process text and speech.",
                "related_concepts": ["text_analysis", "sentiment_analysis", "language_models"],
                "difficulty": "intermediate",
                "multimedia": {
                    "type": "process_flow",
                    "url": "/static/diagrams/nlp_pipeline.svg"
                }
            },
            {
                "topic": "computer_vision",
                "question": "What is Computer Vision?",
                "answer": "Computer Vision is a field of AI that trains computers to interpret and understand visual information from the world. It involves acquiring, processing, analyzing and understanding digital images to extract high-dimensional data from the real world.",
                "related_concepts": ["image_recognition", "convolutional_neural_networks", "object_detection"],
                "difficulty": "intermediate",
                "multimedia": {
                    "type": "demo",
                    "url": "/static/demos/cv_examples.html"
                }
            },
            {
                "topic": "ai_ethics",
                "question": "What are the main ethical considerations in AI?",
                "answer": "Key AI ethics considerations include: 1) Bias and fairness - ensuring AI doesn't discriminate, 2) Transparency and explainability - understanding how AI makes decisions, 3) Privacy and data protection, 4) Accountability for AI decisions, 5) Job displacement concerns, and 6) Safety and security of AI systems.",
                "related_concepts": ["algorithmic_bias", "explainable_ai", "responsible_ai"],
                "difficulty": "advanced",
                "multimedia": {
                    "type": "infographic",
                    "url": "/static/diagrams/ai_ethics_framework.svg"
                }
            },
            {
                "topic": "supervised_learning",
                "question": "What is Supervised Learning?",
                "answer": "Supervised Learning is a type of machine learning where the algorithm learns from labeled training data. The model learns to map inputs to correct outputs by being shown examples of input-output pairs. Common examples include email spam detection and image classification.",
                "related_concepts": ["classification", "regression", "training_data"],
                "difficulty": "intermediate",
                "course_modules": ["ML Fundamentals - Week 2", "Python for Data Science - Module 3"]
            },
            {
                "topic": "unsupervised_learning",
                "question": "What is Unsupervised Learning?",
                "answer": "Unsupervised Learning works with data that has no labeled examples. The algorithm tries to find hidden patterns or structures in the data without being told what to look for. Common techniques include clustering and dimensionality reduction.",
                "related_concepts": ["clustering", "dimensionality_reduction", "pattern_recognition"],
                "difficulty": "intermediate",
                "course_modules": ["Advanced ML Techniques - Week 4"]
            },
            {
                "topic": "deep_learning",
                "question": "What is Deep Learning?",
                "answer": "Deep Learning is a subset of machine learning that uses neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition.",
                "related_concepts": ["neural_networks", "cnn", "rnn", "transformers"],
                "difficulty": "advanced",
                "course_modules": ["Deep Learning Specialization - Week 1-3"]
            },
            {
                "topic": "llms",
                "question": "What are Large Language Models (LLMs)?",
                "answer": "Large Language Models are AI systems trained on vast amounts of text data to understand and generate human-like text. Examples include GPT, BERT, and Claude. They use transformer architecture and can perform various language tasks like translation, summarization, and question-answering.",
                "related_concepts": ["transformers", "attention_mechanism", "pre_training"],
                "difficulty": "advanced",
                "course_modules": ["NLP with Transformers - Week 5-6"]
            },
            {
                "topic": "ai_applications_healthcare",
                "question": "How is AI used in healthcare?",
                "answer": "AI in healthcare includes: medical imaging analysis for disease detection, drug discovery acceleration, personalized treatment recommendations, electronic health record analysis, robotic surgery assistance, and predictive analytics for patient outcomes.",
                "related_concepts": ["medical_imaging", "predictive_analytics", "personalized_medicine"],
                "difficulty": "beginner",
                "real_world_examples": ["IBM Watson for Oncology", "Google DeepMind's protein folding"]
            },
            {
                "topic": "ai_applications_finance",
                "question": "How is AI transforming finance?",
                "answer": "AI in finance powers algorithmic trading, fraud detection, credit scoring, robo-advisors for investment management, chatbots for customer service, and risk assessment. Machine learning models analyze market patterns and customer behavior to make predictions.",
                "related_concepts": ["algorithmic_trading", "fraud_detection", "risk_analysis"],
                "difficulty": "beginner",
                "real_world_examples": ["JPMorgan's COIN", "PayPal's fraud detection"]
            },
            {
                "topic": "ai_applications_autonomous_vehicles",
                "question": "How do autonomous vehicles work?",
                "answer": "Autonomous vehicles use computer vision to see their environment, sensor fusion to combine data from cameras, lidar, and radar, machine learning for decision-making, and real-time processing to navigate safely. They represent a complex integration of multiple AI technologies.",
                "related_concepts": ["computer_vision", "sensor_fusion", "real_time_processing"],
                "difficulty": "advanced",
                "real_world_examples": ["Tesla Autopilot", "Waymo self-driving cars"]
            },
            {
                "topic": "career_paths",
                "question": "What career paths are available in AI?",
                "answer": "AI career paths include: Data Scientist, Machine Learning Engineer, AI Research Scientist, Computer Vision Engineer, NLP Engineer, AI Product Manager, AI Ethics Specialist, and Robotics Engineer. Each requires different skill combinations of programming, mathematics, and domain expertise.",
                "related_concepts": ["data_science", "programming_skills", "mathematics"],
                "difficulty": "beginner",
                "course_modules": ["Career Preparation - Final Module"]
            }
        ]
        
        # Insert knowledge base into MongoDB if not exists
        if knowledge_base.count_documents({}) == 0:
            knowledge_base.insert_many(kb_data)
        
        return kb_data
    
    def define_conversation_flows(self):
        """Define structured conversation flows"""
        return {
            "ai_fundamentals_flow": [
                "What is AI?",
                "How does AI relate to machine learning?",
                "What are some real-world applications?",
                "What should I learn next?"
            ],
            "nlp_exploration": [
                "What is Natural Language Processing?",
                "How do language models work?",
                "What are some NLP applications?",
                "How can I get started with NLP?"
            ],
            "ai_ethics_flow": [
                "What are AI ethics?",
                "What are the main bias concerns?",
                "How can we make AI more transparent?",
                "What's the future of responsible AI?"
            ]
        }
    
    def get_context_aware_response(self, user_message, conversation_context):
        """Generate context-aware responses using OpenAI API"""
        
        # First, check if we have a direct match in knowledge base
        kb_match = self.find_knowledge_base_match(user_message)
        if kb_match:
            return self.format_kb_response(kb_match, conversation_context)
        
        # Generate dynamic response using OpenAI
        system_prompt = self.build_system_prompt(conversation_context)
        
        try:
    # The Gemini API does not use a separate system prompt role in the same way.
    # We will combine it with the user message for a single-turn conversation.
            model = genai.GenerativeModel('gemini-1.5-pro')

            response = model.generate_content(f"{system_prompt}\n\nUser: {user_message}",
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=300,
                    temperature=0.7,
                )
            )

            ai_response = response.text

            # Add follow-up suggestions and related concepts
            enhanced_response = self.enhance_response(ai_response, user_message, conversation_context)

            return enhanced_response

        except Exception as e:
            return {
                "response": "I apologize, but I'm having trouble processing your question right now. Could you try rephrasing it, or ask me about a specific AI topic like machine learning, neural networks, or AI ethics?",
                "type": "fallback",
                "error": str(e)
            }
    
    def find_knowledge_base_match(self, user_message):
        """Find matching content in knowledge base"""
        user_message_lower = user_message.lower()
        
        # Simple keyword matching - can be enhanced with semantic search
        keywords_to_topics = {
            "artificial intelligence": "ai_fundamentals",
            "machine learning": "machine_learning",
            "neural network": "neural_networks",
            "deep learning": "deep_learning",
            "nlp": "nlp",
            "natural language": "nlp",
            "computer vision": "computer_vision",
            "ethics": "ai_ethics",
            "bias": "ai_ethics",
            "supervised learning": "supervised_learning",
            "unsupervised learning": "unsupervised_learning",
            "healthcare": "ai_applications_healthcare",
            "finance": "ai_applications_finance",
            "autonomous": "ai_applications_autonomous_vehicles",
            "career": "career_paths",
            "difference": "ai_vs_ml_vs_dl"
        }
        
        for keyword, topic in keywords_to_topics.items():
            if keyword in user_message_lower:
                return next((item for item in self.knowledge_base if item["topic"] == topic), None)
        
        return None
    
    def format_kb_response(self, kb_item, context):
        """Format knowledge base response with enhancements"""
        response = {
            "response": kb_item["answer"],
            "type": "knowledge_base",
            "topic": kb_item["topic"],
            "difficulty": kb_item["difficulty"],
            "related_concepts": kb_item.get("related_concepts", []),
            "multimedia": kb_item.get("multimedia"),
            "course_modules": kb_item.get("course_modules", []),
            "follow_ups": self.generate_follow_ups(kb_item, context),
            "real_world_examples": kb_item.get("real_world_examples", [])
        }
        
        return response
    
    def generate_follow_ups(self, kb_item, context):
        """Generate contextual follow-up questions"""
        topic = kb_item["topic"]
        follow_ups = []
        
        if topic == "ai_fundamentals":
            follow_ups = [
                "How does AI differ from traditional programming?",
                "What are the types of AI?",
                "Show me some AI applications"
            ]
        elif topic == "machine_learning":
            follow_ups = [
                "What's the difference between supervised and unsupervised learning?",
                "How do I choose the right ML algorithm?",
                "What programming languages are used in ML?"
            ]
        elif topic == "neural_networks":
            follow_ups = [
                "How do neural networks learn?",
                "What are different types of neural networks?",
                "Can you explain backpropagation?"
            ]
        elif topic == "ai_ethics":
            follow_ups = [
                "How can we detect bias in AI systems?",
                "What is explainable AI?",
                "What are AI governance frameworks?"
            ]
        
        return follow_ups[:3]  # Limit to 3 follow-ups
    
    def build_system_prompt(self, context):
        """Build system prompt for OpenAI based on conversation context"""
        base_prompt = """You are an AI educational assistant helping students learn about artificial intelligence, machine learning, and related topics. 

        Your role:
        - Provide clear, accurate explanations of AI concepts
        - Adapt your language to the user's level (beginner, intermediate, advanced)
        - Always suggest related topics they might find interesting
        - Recommend specific learning resources when appropriate
        - Keep responses concise but informative (under 300 words)

        Topics you excel at:
        - AI fundamentals and history
        - Machine learning algorithms and techniques
        - Neural networks and deep learning
        - Natural language processing
        - Computer vision
        - AI ethics and responsible AI
        - Real-world AI applications
        - Career guidance in AI/ML

        Always end your response with a question to keep the conversation going.
        """
        
        if context.get("previous_topics"):
            base_prompt += f"\n\nContext: The user has previously discussed: {', '.join(context['previous_topics'])}"
        
        if context.get("user_level"):
            base_prompt += f"\n\nUser level: {context['user_level']} - adjust your explanation accordingly."
            
        return base_prompt
    
    def enhance_response(self, ai_response, user_message, context):
        """Enhance AI response with additional features"""
        # Extract potential topics mentioned in response for cross-linking
        response_topics = self.extract_topics_from_text(ai_response)
        
        enhanced = {
            "response": ai_response,
            "type": "ai_generated",
            "mentioned_topics": response_topics,
            "suggested_learning_path": self.suggest_learning_path(user_message, context),
            "follow_ups": ["Can you explain this in more detail?", "What are some examples?", "How is this used in practice?"]
        }
        
        return enhanced
    
    def extract_topics_from_text(self, text):
        """Extract AI topics mentioned in the text"""
        text_lower = text.lower()
        mentioned_topics = []
        
        topic_keywords = {
            "machine learning": "machine_learning",
            "neural network": "neural_networks", 
            "deep learning": "deep_learning",
            "computer vision": "computer_vision",
            "natural language": "nlp",
            "ethics": "ai_ethics"
        }
        
        for keyword, topic in topic_keywords.items():
            if keyword in text_lower:
                mentioned_topics.append(topic)
        
        return mentioned_topics
    
    def suggest_learning_path(self, user_message, context):
        """Suggest personalized learning path"""
        user_level = context.get("user_level", "beginner")
        current_topic = context.get("current_topic", "ai_fundamentals")
        
        learning_paths = {
            "beginner": [
                "AI Fundamentals - Week 1",
                "Introduction to Python - Module 1", 
                "Basic Statistics - Module 2",
                "Machine Learning Basics - Week 2"
            ],
            "intermediate": [
                "Advanced ML Algorithms - Week 3",
                "Deep Learning Fundamentals - Week 4",
                "NLP with Python - Module 5"
            ],
            "advanced": [
                "Advanced Deep Learning - Week 6",
                "AI Research Methods - Module 7",
                "MLOps and Production - Week 8"
            ]
        }
        
        return learning_paths.get(user_level, learning_paths["beginner"])

# Initialize chatbot
chatbot = AIEducationChatbot()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', '')
        user_id = data.get('user_id', 'anonymous')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get conversation context
        context = get_conversation_context(session_id)
        
        # Generate response
        response_data = chatbot.get_context_aware_response(user_message, context)
        
        # Save conversation
        save_conversation(session_id, user_id, user_message, response_data)
        
        # Update context
        update_conversation_context(session_id, user_message, response_data)
        
        return jsonify({
            "response": response_data["response"],
            "type": response_data.get("type", "general"),
            "multimedia": response_data.get("multimedia"),
            "follow_ups": response_data.get("follow_ups", []),
            "related_concepts": response_data.get("related_concepts", []),
            "course_modules": response_data.get("course_modules", []),
            "suggested_learning_path": response_data.get("suggested_learning_path", []),
            "session_id": session_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation-flows', methods=['GET'])
def get_conversation_flows():
    """Get available conversation flows"""
    return jsonify({
        "flows": [
            {
                "id": "ai_fundamentals_flow",
                "title": "Explore AI Fundamentals", 
                "description": "Learn the basics of artificial intelligence",
                "duration": "10-15 minutes",
                "difficulty": "beginner"
            },
            {
                "id": "nlp_exploration",
                "title": "Natural Language Processing Journey",
                "description": "Discover how computers understand human language",
                "duration": "15-20 minutes", 
                "difficulty": "intermediate"
            },
            {
                "id": "ai_ethics_flow",
                "title": "AI Ethics & Responsibility",
                "description": "Explore ethical considerations in AI development",
                "duration": "12-18 minutes",
                "difficulty": "advanced"
            }
        ]
    })

@app.route('/api/start-flow/<flow_id>', methods=['POST'])
def start_conversation_flow(flow_id):
    """Start a structured conversation flow"""
    data = request.get_json()
    session_id = data.get('session_id', '')
    
    flows = chatbot.conversation_flows
    if flow_id not in flows:
        return jsonify({"error": "Flow not found"}), 404
    
    flow_questions = flows[flow_id]
    first_question = flow_questions[0]
    
    # Initialize flow context
    flow_context = {
        "flow_id": flow_id,
        "current_step": 0,
        "total_steps": len(flow_questions),
        "questions": flow_questions
    }
    
    # Save flow context
    conversations.update_one(
        {"session_id": session_id},
        {"$set": {"flow_context": flow_context}},
        upsert=True
    )
    
    # Get response for first question
    context = get_conversation_context(session_id)
    response_data = chatbot.get_context_aware_response(first_question, context)
    
    return jsonify({
        "message": f"Starting {flow_id.replace('_', ' ').title()}!",
        "question": first_question,
        "response": response_data["response"],
        "flow_progress": {
            "current_step": 1,
            "total_steps": len(flow_questions),
            "progress_percentage": round(100 / len(flow_questions))
        },
        "next_question": flow_questions[1] if len(flow_questions) > 1 else None
    })

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    data = request.get_json()
    
    feedback_data = {
        "session_id": data.get('session_id'),
        "rating": data.get('rating'), 
        "comment": data.get('comment', ''),
        "helpful": data.get('helpful'), 
        "suggestion": data.get('suggestion', ''),
        "timestamp": datetime.utcnow()
    }
    
    db.feedback.insert_one(feedback_data)
    
    return jsonify({"message": "Feedback received. Thank you!"})

@app.route('/api/progress/<user_id>', methods=['GET'])
def get_user_progress(user_id):
    """Get user learning progress"""
    progress = user_progress.find_one({"user_id": user_id})
    
    if not progress:
        return jsonify({
            "topics_covered": [],
            "total_conversations": 0,
            "learning_level": "beginner",
            "recommended_next_topics": ["ai_fundamentals", "machine_learning"]
        })
    
    return jsonify(progress)

@app.route('/api/learning-path/<user_id>', methods=['GET'])
def get_learning_path_recommendation(user_id):
    """Get a personalized learning path recommendation"""
    # Get user context (you may need to extend this)
    context = get_conversation_context(user_id) 

    recommended_path = chatbot.suggest_learning_path(None, context)

    return jsonify({
        "user_id": user_id,
        "recommended_path": recommended_path
    })


def get_conversation_context(session_id):
    """Get conversation context from database"""
    context = conversations.find_one({"session_id": session_id})
    
    if not context:
        return {
            "previous_topics": [],
            "user_level": "beginner",
            "message_count": 0,
            "current_topic": None
        }
    
    return {
        "previous_topics": context.get("topics", []),
        "user_level": context.get("user_level", "beginner"),
        "message_count": context.get("message_count", 0),
        "current_topic": context.get("current_topic"),
        "flow_context": context.get("flow_context")
    }

def save_conversation(session_id, user_id, user_message, bot_response):
    """Save conversation to database"""
    conversation_entry = {
        "session_id": session_id,
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        "user_message": user_message,
        "bot_response": bot_response,
        "response_type": bot_response.get("type", "general")
    }
    
    conversations.update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": conversation_entry},
            "$inc": {"message_count": 1},
            "$set": {"last_activity": datetime.utcnow()}
        },
        upsert=True
    )

def update_conversation_context(session_id, user_message, bot_response):
    """Update conversation context"""
    # Extract topic from response
    current_topic = bot_response.get("topic")
    mentioned_topics = bot_response.get("mentioned_topics", [])

    update_doc = {}
    if current_topic:
        update_doc["$set"] = {"current_topic": current_topic}

    if current_topic or mentioned_topics:
        if "$addToSet" not in update_doc:
            update_doc["$addToSet"] = {}
        update_doc["$addToSet"]["topics"] = {"$each": [current_topic] + mentioned_topics}

    if update_doc:
        conversations.update_one(
            {"session_id": session_id},
            update_doc,
            upsert=True
        )

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)