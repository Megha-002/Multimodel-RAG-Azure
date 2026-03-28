import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")

# Create speech configuration
speech_config = speechsdk.SpeechConfig(
    subscription=speech_key,
    region=speech_region
)

# Use default microphone
speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config
)

print("🎤 Speak now...")

# Recognize speech
result = speech_recognizer.recognize_once()

# Handle result
if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("\n✅ Recognized Text:")
    print(result.text)

elif result.reason == speechsdk.ResultReason.NoMatch:
    print("\n❌ No speech could be recognized")

elif result.reason == speechsdk.ResultReason.Canceled:
    print("\n❌ Speech recognition canceled")
    cancellation_details = result.cancellation_details
    print("Reason:", cancellation_details.reason)
    print("Error details:", cancellation_details.error_details)
