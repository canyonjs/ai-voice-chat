import speech_recognition as sr
import openai
import requests
from dotenv import load_dotenv
from os import getenv
from elevenlabs import set_api_key, Voice, VoiceSettings, generate, play

load_dotenv()

openai.api_key = getenv("OPENAI_API_KEY")
set_api_key(getenv("ELEVENLABS_API_KEY"))

recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Elevenlabs Config
elevenlabs_dorothy = "ThT5KcBeYPX3keUQqHPh"



def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }
    
    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio, language="en-US")
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response

if __name__ == "__main__":
    for j in range(3):
        print("Speak prompt now...")
    
        speech = recognize_speech_from_mic(recognizer, microphone)
    
        if speech["transcription"]:
            break
        if not speech["success"]:
            break
        print("I didn't catch that. What did you say?\n")

    prompt = speech["transcription"]

    print("Prompt:", prompt)

    print("Sending to OpenAI...")
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can use other engines like "text-davinci-001" or "text-davinci-003" as well
        prompt=prompt,
        max_tokens=50  # You can adjust this based on the desired response length
    )

    llm_response = response.choices[0].text
    print("Response:", llm_response)

    print("Generating audio...")
    audio = generate(
        text=llm_response,
        voice=Voice(
            voice_id=elevenlabs_dorothy,
            settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
        )
    )
    
    play(audio)
    print(llm_response)