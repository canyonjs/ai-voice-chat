import speech_recognition as sr
import openai
import requests
from dotenv import load_dotenv
from os import getenv
from sys import exit
from elevenlabs import set_api_key, Voice, VoiceSettings, generate, play

load_dotenv()

openai.api_key = getenv("OPENAI_API_KEY")
set_api_key(getenv("ELEVENLABS_API_KEY"))

recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Elevenlabs Config
elevenlabs_dorothy_id = "ThT5KcBeYPX3keUQqHPh"

TRANSCRIBE_RETRY_ATTEMPTS = 3


def transcribe_speech(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        transcription_response = {
            "content": "",
            "error_code": 0
        }

        for j in range(TRANSCRIBE_RETRY_ATTEMPTS):
            print("Listening...")
            try:
                transcription_response.content = recognizer.recognize_google(audio, language="en-US")
                break
            except sr.RequestError:
                transcription_response.content = "Transcription API unavailable"
                transcription_response.error_code = 1
                break
            except sr.UnknownValueError:
                print("I couldn't understand you, please try again.")
                transcription_response.content = "Unrecognized speech"
                transcription_response.error_code = 2
        
        return transcription_response

def query_chatgpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can use other engines like "text-davinci-001" or "text-davinci-003" as well
        prompt=prompt,
        max_tokens=50  # You can adjust this based on the desired response length
    )
    
    return response.choices[0].text

if __name__ == "__main__":
    prompt = transcribe_speech(recognizer, microphone)

    if prompt.error_code != "0":
        print(f"Something went wrong or you exhausted your retry attempts. Error code: {str(prompt.error_code)} | {prompt.content}")
        exit()


    print("Prompt:", prompt)
    print("Sending to ChatGPT...")

    query_response = query_chatgpt(prompt)
    print("Response:", query_response)

    print("Generating audio...")
    audio = generate(
        text=llm_response,
        voice=Voice(
            voice_id=elevenlabs_dorothy_id,
            settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
        )
    )
    
    play(audio)
    print(llm_response)