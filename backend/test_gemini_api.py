import google.generativeai as genai
import os

# Configure with your API key from the .env file
# Be sure the .env file is in the same directory and loaded
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Hello, what is your name?")
        print("Success! API key is working.")
        print("Model response:", response.text)
    except Exception as e:
        print("Failure! The API key is likely the issue.")
        print("Specific error:", e)