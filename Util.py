import os, io
from google.cloud import vision_v1
from google.cloud import language_v1, speech

import time
import pyaudio
import pandas as pd
import math
import wave

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "hackduke-2020-297720-1aecc6f0b1fc.json"


language_client = language_v1.LanguageServiceClient()
speech_client = speech.SpeechClient()

def record():
    p =pyaudio.PyAudio()

    recordTime = 7;
    stream = p.open(format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    input_device_index=1,
    frames_per_buffer=1024)

    frames = []

    for i in range(0, int(16000/1024*recordTime)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = wave.open("TempAudio.wav", "wb")
    audio.setnchannels(1)
    audio.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    audio.setframerate(16000)
    audio.writeframes(b"".join(frames))
    audio.close()



def stream_transcribe(audio_path):
    with io.open(audio_path, "rb") as audio:
        content = audio.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = speech_client.recognize(config=config, audio=audio)

    time = 0.0
    for result in response.results:
        print("Transcript: ", result.alternatives[0].transcript)
        string = result.alternatives[0].transcript
        classify_text(string)


def classify_text(text):
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text, "type_": type_, "language": language}

    response = language_client.classify_text(request={"document": document})
    for category in response.categories:
        print("Category: ", category.name)
        print("Confidence: ", category.confidence)


record()
stream_transcribe("TempAudio.wav")