# ===============================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ===============================

import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# ===============================
# STEP 2: LOAD ENV VARIABLES
# ===============================

load_dotenv()

# Azure Speech
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# Azure OpenAI
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# ===============================
# STEP 3: FUNCTION → VOICE TO TEXT
# ===============================

def get_text_from_voice():
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config
    )

    print("🎤 Speak now...")
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("\n✅ You said:")
        print(result.text)
        return result.text

    else:
        print("❌ Could not recognize speech")
        return None

# ===============================
# STEP 4: FUNCTION → SEND TEXT TO GPT-4
# ===============================

def ask_gpt4(user_text):
    client = AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version=OPENAI_VERSION
    )

    response = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_text}
        ]
    )

    return response.choices[0].message.content

# ===============================
# STEP 5: MAIN FLOW (VOICE → GPT)
# ===============================

if __name__ == "__main__":
    voice_text = get_text_from_voice()

    if voice_text:
        print("\n🤖 Sending to GPT-4...")
        answer = ask_gpt4(voice_text)

        print("\n💡 GPT-4 Answer:\n")
        print(answer)
# meghna git hub