import openai
import os

api_key = os.getenv()
openai.api_key = api_key

def query_llm(audio_file_path):
    try:
        audio_file = open(audio_file_path, "rb")

        # Correct method for GPT-4o (ChatCompletion with audio)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this audio."},
                        {"type": "audio", "audio": audio_file}
                    ]
                }
            ],
            max_tokens=100
        )
        
        return response.choices[0].message["content"]
    
    except Exception as e:
        print(f"Error querying GPT-4o for audio: {e}")
        return "Sorry, I couldn't process that."

