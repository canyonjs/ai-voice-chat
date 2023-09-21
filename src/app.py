import speech_recognition as sr
import openai
import requests
from dotenv import load_dotenv
from os import getenv
from sys import exit
from elevenlabs import set_api_key, Voice, VoiceSettings, generate, play, stream

load_dotenv()


recognizer = sr.Recognizer()
microphone = sr.Microphone()

# General Configuration
openai.api_key = getenv("OPENAI_API_KEY")
TRANSCRIBE_RETRY_ATTEMPTS = 3
OPENAI_ENGINE = "gpt-3.5-turbo"
OPENAI_MAX_RESPONSE_TOKENS = 125
HIDDEN_PROMPT = "Ensure that your response is concise. "


# Elevenlabs Configuration
set_api_key(getenv("ELEVENLABS_API_KEY"))
elevenlabs_voice_id = "ThT5KcBeYPX3keUQqHPh" # Dorothy



def transcribe_speech(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        transcription_response = {
            "content": "",
            "error_code": None
        }

        try:
            transcription_response["content"] = recognizer.recognize_google(audio, language="en-US")
            transcription_response["error_code"] = 0
        except sr.RequestError:
            transcription_response["content"] = "Transcription API unavailable"
            transcription_response["error_code"] = 1
        except sr.UnknownValueError:
            transcription_response["content"] = "I couldn't understand you, please try again."
            transcription_response["error_code"] = 2
        except Exception as err:
            transcription_response["content"] = f"An error has occurred. {err}"
            transcription_response["error_code"] = 3
        
        return transcription_response

def query_chatgpt(prompt):
    # response = openai.Completion.create(
    #     engine=OPENAI_ENGINE,  # You can use other engines like "text-davinci-001" or "text-davinci-003" as well
    #     max_tokens=OPENAI_MAX_RESPONSE_TOKENS, # You can adjust this based on the desired response length
    #     prompt=prompt
    # )

    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": HIDDEN_PROMPT + prompt}],
        stream=True,
        temperature=0,
        max_tokens=OPENAI_MAX_RESPONSE_TOKENS
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            yield content

def synthesize_speech(text_stream):
    audio_stream = generate(
        text=text_stream,
        voice="Nicole",
        model="eleven_monolingual_v1",
        stream=True
    )

    return audio_stream


if __name__ == "__main__":
    for attempt in range(TRANSCRIBE_RETRY_ATTEMPTS):
        print("Listening...")
        transcription_response = transcribe_speech(recognizer, microphone)
        
        if transcription_response["error_code"] == 0:
            prompt = transcription_response["content"]
            print(f"Prompt: {prompt}")
            break
        elif transcription_response["error_code"] == 2:
            print(f"{transcription_response['content']} | Error code: {str(transcription_response['error_code'])}")

            if attempt == TRANSCRIBE_RETRY_ATTEMPTS - 1:
                print("Speech transcription retries exhausted.")
                exit()
        elif transcription_response["error_code"] == 1 or transcription_response["error_code"] == 3:
            print(f"{transcription_response['content']} | Error code: {str(transcription_response['error_code'])}")
            exit()
   

    print("Querying ChatGPT and synthesizing via ElevenLabs...")

    stream(synthesize_speech(query_chatgpt(prompt)))

