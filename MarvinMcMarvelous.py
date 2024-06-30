#!/usr/bin/env -S python -u
#############################################################################

import argparse
import json
import logging
import re
import sys

from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import torch

# just for the alert sound
import numpy as np

import base64
import io
import pyaudio
import wave

import requests
import json

import nltk

############################################
import tkinter as tk
from tkinter import Label, Button, filedialog
from PIL import Image, ImageTk
import threading
############################################

##
 #
 # Welcome to MarvinMcMarvelous, a local speech-to-text budro!
 #
 ##
class MarvinMcMarvelous:
    LOG = logging.getLogger(__name__)
    JLLM = json.dumps({
        "prompt": "<<prompt>>",
        "model": "phi3",
        "stream": False,
        "system": "You are a helpful AI assistant. Your responses are read aloud so you strive for brevity and don't include code snippets or disclaimers.",
        "options": {
            "seed": 101,
            "temperature": 0
        }
    })
    JTTS = json.dumps({"text": "<<prompt>>", "voice": "en_US-amy-medium"})
    JTTI = json.dumps({"prompt": "<<prompt>>"})

    def __init__(self):
        self.tk_root = None
        self.tk_thread = None
        self.parser = argparse.ArgumentParser(description="Marvin McMarvelous is secretly in love with you!!!")
        self.parser.add_argument("--use_cuda"              , action="store_false", default=True)
        self.parser.add_argument("--quiet"                 , action="store_true", default=False)
        self.parser.add_argument("--wake_debug"            , action="store_true",  default=False)
        self.parser.add_argument("--chop"                  , action="store_true",  default=False)
        self.parser.add_argument("--wake_model"            , type=str,   default="MIT/ast-finetuned-speech-commands-v2")
        self.parser.add_argument("--wake_word"             , type=str,   default="marvin")
        self.parser.add_argument("--wake_prob_threshold"   , type=float, default=0.5)
        self.parser.add_argument("--wake_chunk_length_s"   , type=float, default=2.0)
        self.parser.add_argument("--wake_stream_chunk_s"   , type=float, default=0.25)
        self.parser.add_argument("--listen_model"          , type=str,   default="openai/whisper-base.en")
        self.parser.add_argument("--listen_chunk_length_s" , type=float, default=7.0)
        self.parser.add_argument("--listen_stream_chunk_s" , type=float, default=1.0)
        self.parser.add_argument("--listen_tokens"         , type=int,   default=128)
        self.parser.add_argument("--exit_regex"            , type=str,   default="^exit loop")
        self.parser.add_argument("--repeat_regex"          , type=str,   default="^repeat\W*$")
        self.parser.add_argument("--exit_message"          , type=str,   default="See you later, space cowboy")
        self.parser.add_argument("--llm_url"               , type=str,   default="http://aid:11434/api/generate")
        self.parser.add_argument("--llm_json"              , type=str,   default=MarvinMcMarvelous.JLLM)
        self.parser.add_argument("--tts_url"               , type=str,   default="http://aid:5000/")
        self.parser.add_argument("--tts_json"              , type=str,   default=MarvinMcMarvelous.JTTS)
        self.parser.add_argument("--tti_url"               , type=str,   default="http://aid:7860/sdapi/v1/txt2img")
        self.parser.add_argument("--tti_json"              , type=str,   default=MarvinMcMarvelous.JTTI)
    # end of __init__

    def main(self):
        args = self.args = self.parser.parse_args()
        MarvinMcMarvelous.LOG.info(f'settings: {json.dumps(vars(args))}')
        device = "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"

        try:
            with open("mcm.png", "rb") as file:
                self.showImage(file.read())
        except Exception as error:
            print(f"\033[31m🙊 Could not load welcome image\033[0m")

        self.classifier = pipeline(
            "audio-classification", 
            model=args.wake_model,
            device=device
        )

        self.transcriber = pipeline(
            "automatic-speech-recognition", 
            model=args.listen_model,
            device=device
        )

        exit_regex = re.compile(args.exit_regex, re.IGNORECASE)
        repeat_regex = re.compile(args.repeat_regex, re.IGNORECASE)
        print(f"\033[32m🏃🚪Exit pattern is \033[31m{args.exit_regex}\033[0m", file=sys.stderr)

        if not args.quiet:
            self.speakUp(f"The alert word is {args.wake_word}. The exit pattern is {args.exit_regex}. The repeat pattern is {args.repeat_regex}.")

        prior = False

        while True: 
            ok = self.wake_up(
                wake_word=args.wake_word,
                prob_threshold=args.wake_prob_threshold,
                chunk_length_s=args.wake_chunk_length_s,
                stream_chunk_s=args.wake_stream_chunk_s,
                debug=args.wake_debug,
            )
            if not ok:
                print(f"\033[33mExiting due to error listening for wake word.\033[0m", file=sys.stderr)
                break

            speech = self.transcribe(
                chunk_length_s=args.listen_chunk_length_s,
                stream_chunk_s=args.listen_stream_chunk_s,
                max_new_tokens=args.listen_tokens,
            )
            if speech is None:
                print(f"\033[33mExiting due to error listening for speech.\033[0m", file=sys.stderr)
                break

            if re.match(exit_regex, speech):
                print(f"exiting, on: '{speech}'", file=sys.stderr)
                break
            if prior and re.match(repeat_regex, speech):
                print(f"repeating", file=sys.stderr)
                speech = prior

            print(">", speech, file=sys.stderr)
            prior = speech
            self.onText(speech)

        self.onExit()
    # end of main

    def wake_up(
        self,
        wake_word="marvin",
        prob_threshold=0.5,
        chunk_length_s=2.0,
        stream_chunk_s=0.25,
        debug=False,
    ):
        if wake_word not in self.classifier.model.config.label2id.keys():
            raise ValueError(
                f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {self.classifier.model.config.label2id.keys()}."
            )

        sampling_rate = self.classifier.feature_extractor.sampling_rate

        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )

        if True or debug:
            print(f"\033[32m🧏 Listening for wake word 👀: \033[33m{wake_word}\033[0m", file=sys.stderr)

        for prediction in self.classifier(mic):
            prediction = prediction[0]
            if debug:
                print(prediction, file=sys.stderr)
            if prediction["label"] == wake_word:
                if prediction["score"] > prob_threshold:
                    return True
        return False
    # end of wake_up

    def transcribe(self, chunk_length_s=5.0, stream_chunk_s=1.0, max_new_tokens=128):
        sampling_rate = self.transcriber.feature_extractor.sampling_rate

        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )

        print("\n\n\033[32m🤖👂Hey! I'm listening to you now...\033[0m", file=sys.stderr)
        try:
            self.alert()
        except Exception as error:
            print("....", file=sys.stderr)

        item = None

        for item in self.transcriber(mic, generate_kwargs={"max_new_tokens": max_new_tokens}):
            sys.stderr.write("\033[K")
            print(item["text"], end="\r", file=sys.stderr)
            if not item["partial"][0]:
                break

        if item is None:
            return item

        return item["text"].strip()
    # end of transcribe

    def onText(self, text):
        url = self.args.llm_url
        data = json.loads(self.args.llm_json)
        data = self.recurseReplace(data, text)
        self.onTextResponse(self.post_it(url, data))
    # end of onText

    def onTextResponse(self, response):
        data = response.json()
        text = data["response"] # FIXME: this is not generic 
        print('onTextResponse:', text, file=sys.stderr)
        self.speakUp(text, True)
    # end of onTextResponse

    def speakUp(self, text, isRobot=False):
        url = self.args.tti_url
        data = json.loads(self.args.tti_json)
        data = self.recurseReplace(data, text)
        self.onImage(self.post_it(url, data))

        if isRobot and self.args.chop:
            for t in nltk.sent_tokenize(text):
                text = t
                print('chop:', text, file=sys.stderr)
                break
                
        url = self.args.tts_url
        data = json.loads(self.args.tts_json)
        data = self.recurseReplace(data, text)
        self.onSpeech(self.post_it(url, data))
    # end of speakUp
    
    def onSpeech(self, response):
        print('onSpeech', response, file=sys.stderr)
        if response is None:
            print("\033[31mTTS FAILED!!!\033[0m", file=sys.stderr)
            self.alert(44.44)
            return
        data = response.json()
        audio = data['audio'] # FIXME: this is not generic 
        decoded = base64.b64decode(audio)
        wav_file = io.BytesIO(decoded)
        wav = wave.open(wav_file, 'rb')
        self.playWav(wav)
    # end of onSpeech

    def playWav(self, wav):
        p = pyaudio.PyAudio()
        s = p.open(
            format = p.get_format_from_width(wav.getsampwidth()), 
            channels = wav.getnchannels(), 
            rate = wav.getframerate(),
            output = True
        )

        chunk = 1024 * 16
        data = wav.readframes(chunk)
        while data != b'':
          s.write(data)
          data =  wav.readframes(chunk)
        s.close()
        p.terminate()
    # end of playWav

    def post_it(self, url, request):
        body = json.dumps(request)
        headers = {'Content-Type': 'application/json'}
        response = None
        try:
            response = requests.post(url, data=body, headers=headers)
            if response.status_code < 200 or response.status_code > 299:
                print('oops!', response, file=sys.stderr)
                response = None
        except Exception as e:
            print('ouch', e, file=sys.stderr)
        return response
    # end of post_it

    def alert(self, frequency=261.63, duration=.25, volume=.11, sampling_rate=44100):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=sampling_rate, output=True)
        for f in [frequency * 8, frequency * 4]:
            t = np.linspace(0, duration, int(duration * sampling_rate), False)
            note = np.sin(f * 2 * np.pi * t) * volume
            note_bytes = note.astype(np.float32).tobytes()
            stream.write(note_bytes)
        stream.stop_stream()
        stream.close()
    # end of alert

    def recurseReplace(self, d, text, template = "<<prompt>>"):
        if (isinstance(d, list)):
            for i,v in enumerate(d):
                d[i] = self.recurseReplace(v, text, template)
        else:
            if (isinstance(d, dict)):
                for k,v in d.items():
                    d[k] = self.recurseReplace(v, text, template)
            else:
                if (isinstance(d, str)):
                    d = d.replace(template, text)
        return d
    # end of recurseReplace

    def onImage(self, response):
        data = response.json()
        image_data = base64.b64decode(data["images"][0]) # FIXME: this is not generic 
        self.showImage(image_data)
    # end of onImage

    def showImage(self, image_data):
        image = Image.open(io.BytesIO(image_data))
        if self.tk_root is None:
            self.setupTkLoop(image)
        else:
            self.photo = ImageTk.PhotoImage(image)
            self.label.config(image=self.photo)
            self.label.image = self.photo
    # end of showImage

    def setupTkLoop(self, image):
        def run_tk():
            self.tk_root = tk.Tk()
            self.tk_root.title("MarvinMcMarvelous's Image Viewer")
            self.photo = ImageTk.PhotoImage(image)
            self.label = Label(self.tk_root, image=self.photo)
            self.label.pack()
            self.save_button = Button(self.tk_root, text="Save Image", command=self.save_image)
            self.save_button.pack()
            self.tk_root.mainloop()
            self.tk_root.destroy()
        self.tk_thread = threading.Thread(target=run_tk)
        self.tk_thread.start()
    # end of setupTkLoop

    def save_image(self):
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes)
        if file_path:
            self.photo._PhotoImage__photo.write(file_path)

    def onExit(self):
        self.speakUp(self.args.exit_message)
        if self.tk_thread:
            print(f"\033[31m ⏻  \033[32mShutting down the TK window, be patient or close it manually\033[0m", file=sys.stderr)
            self.tk_root.quit()
            self.tk_thread.join()
            print(f"\033[32m🙊 Sorry, it still dumps core and idk why...\033[0m", file=sys.stderr)
    # enf of onExit

# end of class MarvinMcMarvelous

if __name__ == "__main__":
    MarvinMcMarvelous().main()

# EOF
#############################################################################
