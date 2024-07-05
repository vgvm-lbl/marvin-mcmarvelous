#!/usr/bin/env -S python -u
#############################################################################

import argparse
import json
import logging
import os
import re
import sys
import time

from typing import Any

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

from SystemPrompts import System

############################################
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import Label, Button, Text, Scrollbar, VERTICAL, RIGHT, Y, Frame, filedialog, messagebox
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
        "model": "llama3",
        #"model": "phi3",
        "stream": False,
        "system": "You are a helpful AI assistant. Your responses are read aloud so you strive for brevity and don't include code snippets or disclaimers.",
        "options": {
            "seed": 101,
            "temperature": 0
        }
    })
    JTTS = json.dumps({"text": "<<prompt>>", "voice": "en_US-joe-medium"})
    JTTI = json.dumps({"prompt": "<<prompt>>"})

    def __init__(self):
        self.tk_root = None
        self.bot_thread = None
        self.lastText = None
        self.parser = argparse.ArgumentParser(description="Marvin McMarvelous is secretly in love with you!!!")
        self.parser.add_argument("--use_cuda"              , action="store_false", default=True)
        self.parser.add_argument("--quiet"                 , action="store_true", default=False)
        self.parser.add_argument("--wake_debug"            , action="store_true",  default=False)
        self.parser.add_argument("--chop"                  , action="store_true",  default=False)
        self.parser.add_argument("--wake_model"            , type=str,   default="MIT/ast-finetuned-speech-commands-v2")
        self.parser.add_argument("--wake_words"            , type=str,   default="marvin")
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
        self.parser.add_argument("--system"                , type=str,   default="")
        self.parser.add_argument("--load"                  , type=str,   default="")
    # end of __init__

    def main(self):
        self.loadArgs()
        MarvinMcMarvelous.LOG.info(f'settings: {json.dumps(vars(self.args))}')

        # start the bot thread
        def runBots():
            self._main()
        self.bot_thread = threading.Thread(target=runBots)
        self.bot_thread.start()

        with open("mcm.png", "rb") as file:
            self.showImage(file.read())
    # end of main

    def loadArgs(self):
        a = self.args = self.parser.parse_args()
        if not a.load:
            return a
        loaded = self.load_json(a.load)
        for k, v in loaded.items():
            if (hasattr(a, k)):
                setattr(a, k, v)
                continue
            j = f"{k}_json"
            if (hasattr(a, j)):
                setattr(a, j, json.dumps(v))
        print("-----------------------------------------------------------------") 
        return self.args
    # end of loader

    def _main(self):
        device = "cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu"

        self.classifier = pipeline(
            "audio-classification", 
            model=self.args.wake_model,
            device=device
        )

        self.transcriber = pipeline(
            "automatic-speech-recognition", 
            model=self.args.listen_model,
            device=device
        )

        exit_regex = re.compile(self.args.exit_regex, re.IGNORECASE)
        repeat_regex = re.compile(self.args.repeat_regex, re.IGNORECASE)
        print(f"\033[32m🏃🚪Exit pattern is \033[31m{self.args.exit_regex}\033[0m", file=sys.stderr)

        wake_words = [w.strip() for w in self.args.wake_words.split(",")]

        # see if the system value is a named system
        from_system = System.get(self.args.system)
        if from_system:
            #wake_words[1] = wake_words[0] # disable dynamic system prompt
            print(f"⚙️ Using system: '\033[31m{self.args.system}\033[0m'")
            self.args.system = from_system

        if self.args.load:
            #wake_words[1] = wake_words[0] # disable dynamic system prompt
            print(f"🌟 Using personality: '\033[33m{self.args.load}\033[0m'")

        wake_word = wake_words[0]
        welcome = f"The alert word is {wake_word}. The exit pattern is {self.args.exit_regex}. The repeat pattern is {self.args.repeat_regex}."
        if self.args.quiet:
            self.update_text_area(welcome)
        else:
            self.speakUp(welcome)

        prior = False

        while True: 
            woke = self.wake_up(
                wake_words=wake_words,
                #wake_word=self.args.wake_word,
                prob_threshold=self.args.wake_prob_threshold,
                chunk_length_s=self.args.wake_chunk_length_s,
                stream_chunk_s=self.args.wake_stream_chunk_s,
                debug=self.args.wake_debug,
            )
            if not woke:
                print(f"\033[33mExiting due to error listening for wake word.\033[0m", file=sys.stderr)
                break

            setting_system = 1 == wake_words.index(woke)
            chunk_length_s = self.args.listen_chunk_length_s
            if setting_system:
                chunk_length_s = 20
                self.speakUp("Set the LLM system prompt")

            speech = self.transcribe(
                chunk_length_s=chunk_length_s,
                stream_chunk_s=self.args.listen_stream_chunk_s,
                max_new_tokens=self.args.listen_tokens,
            )
            if speech is None:
                print(f"\033[33mExiting due to error listening for speech.\033[0m", file=sys.stderr)
                break

            if setting_system:
                print("--------------------------------------------------------------", file=sys.stderr)
                # this is pretty hacky
                data = json.loads(self.args.llm_json)
                if "system" in data:
                    data["system"] = speech
                    self.args.llm_json = json.dumps(data)
                    self.speakUp(f"Setting LLM system prompt to {speech}")
                print("--------------------------------------------------------------", file=sys.stderr)
                continue

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
        wake_words=["marvin", "learn"],
        prob_threshold=0.5,
        chunk_length_s=2.0,
        stream_chunk_s=0.25,
        debug=False,
    ):
        for wake_word in wake_words:
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
            print(f"\033[32m🧏 Listening for wake word 👀: \033[33m{wake_words}\033[0m", file=sys.stderr)

        for prediction in self.classifier(mic):
            prediction = prediction[0]
            if debug:
                print(prediction, file=sys.stderr)
            for wake_word in wake_words:
                if prediction["label"] == wake_word:
                    if prediction["score"] > prob_threshold:
                        return wake_word
        return None
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

    def toDict(self, prompt):
        return {
            "prompt": prompt,
            "now": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    # end of toDict

    def onText(self, text):
        url = self.args.llm_url
        data = json.loads(self.args.llm_json)
        data = self.recurseReplace(data, self.toDict(text))
        if self.args.system and "system" in data:
            data["system"] = self.args.system
        self.update_text_area(f"You said: {text}")
        self.onTextResponse(self.post_it(url, data))
    # end of onText

    def onTextResponse(self, response):
        data = response.json()
        text = data["response"] # FIXME: this is not generic 
        print('onTextResponse:', text, file=sys.stderr)
        self.speakUp(text, True)
        self.lastText = text
    # end of onTextResponse

    def speakUp(self, text, isRobot=False):
        self.update_text_area(text)
        def run_tti():
            self.runTti(text)

        tti_thread = threading.Thread(target=run_tti)
        tti_thread.start()

        if isRobot and self.args.chop:
            for t in nltk.sent_tokenize(text):
                text = t
                print('chop:', text, file=sys.stderr)
                break
                
        url = self.args.tts_url
        data = json.loads(self.args.tts_json)
        data = self.recurseReplace(data, self.toDict(text))
        self.onSpeech(self.post_it(url, data))

        tti_thread.join()
    # end of speakUp

    def runTti(self, text: str):
        self.set_busy_status(True)
        url = self.args.tti_url
        data = json.loads(self.args.tti_json)
        data = self.recurseReplace(data, self.toDict(text))
        self.onImage(self.post_it(url, data))
        self.set_busy_status(False)
    # end of runTti

    
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

    def onImage(self, response):
        data = response.json()
        image_data = base64.b64decode(data["images"][0]) # FIXME: this is not generic 
        self.showImage(image_data)
    # end of onImage

    def showImage(self, image_data):
        image = Image.open(io.BytesIO(image_data))
        if self.tk_root is None:
            self.runTkLoop(image)
        else:
            self.photo = ImageTk.PhotoImage(image)
            self.label.config(image=self.photo)
            self.label.image = self.photo
    # end of showImage

    def setupTk(self, image):
        self.tk_root = tk.Tk()
        self.tk = self.tk_root # ???
        self.tk_root.title("MarvinMcMarvelous's Image Viewer")

        # Create a frame to hold the image and text area
        frame = Frame(self.tk_root)
        frame.pack()

        # Display the image
        self.photo = ImageTk.PhotoImage(image)
        self.label = Label(frame, image=self.photo)
        self.label.pack(side="left", padx=10, pady=10)

        # Create a text area for displaying text
        self.text_area = Text(frame, wrap="word", height=33, width=66)
        self.text_area.insert("1.0", "Loading models, please be patient!")
        self.text_area.configure(state="disabled")
        self.text_area.pack(side="left", padx=10, pady=10)

        if False:
            scrollb = ttk.Scrollbar(self, command=self.text_area.yview)
            scrollb.grid(row=0, column=1, sticky='nsew')
            self.text_area['yscrollcommand'] = scrollb.set

        buttons = Frame(self.tk_root)
        buttons.pack()

        # Add a button to save the image
        self.save_button = Button(buttons, text="Save Image", command=self.save_image)
        self.save_button.pack(side="left")

        # Add a button to save the image
        self.save_button = Button(buttons, text="Reroll Image", command=self.reroll_image)
        self.save_button.pack(side="left")

        # Add a label to show busy status
        self.status_label = Label(buttons, text="👦", fg="red")
        self.status_label.pack(side="left")

    def set_busy_status(self, is_busy):
        if is_busy:
            self.status_label.config(text="🏃")
        else:
            self.status_label.config(text="")
    # end of set_busy_status

    def update_text_area(self, new_text):
        self.text_area.configure(state="normal")
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert("1.0", new_text)
        self.text_area.configure(state="disabled")
    # end of update_text_area

    def runTkLoop(self, image):
        self.setupTk(image)
        self.tk_root.mainloop()
        self.tk_root.destroy()
    # end of runTkLoop

    def save_image(self):
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes)
        if file_path:
            self.photo._PhotoImage__photo.write(file_path)
    # end of save_image

    def reroll_image(self):
        if self.lastText:
            self.runTti(self.lastText)
        else:
            print("FIXME: that button should be disabled till now...")
    # end of reroll_image

    def onExit(self):
        if not self.args.quiet:
            self.onText(self.args.exit_message)
        self.speakUp(self.args.exit_message)
        print(f"\033[31m⏻  \033[32mShutting down the TK window, be patient or close it manually\033[0m", file=sys.stderr)
        self.tk_root.quit()
        #if self.bot_thread:
        #    self.bot_thread.join()
    # end of onExit

    def load_json(self, filename: str, quiet: bool = True) -> Any:
       if not os.path.exists(filename):
           if not quiet:
               print("file not found", filename)
           return {}
       with open(filename, 'r') as file:
           return json.load(file)
    # end of load_json

    def recurseReplace(self, d, theDict:dict):
        if (isinstance(d, list)):
            for i,v in enumerate(d):
                d[i] = self.recurseReplace(v, theDict)
        else:
            if (isinstance(d, dict)):
                for k,v in d.items():
                    d[k] = self.recurseReplace(v, theDict)
            else:
                if (isinstance(d, str)):
                    #d = d.replace(template, text)
                    d = self.debracker(d, theDict)
        return d
    # end of recurseReplace

    # seach for <<key>> and replace it with the value from the dict
    def debracker(self, text:str, d:dict) -> str:
        x = re.compile(r'<<([^>]+)>>')
        def replace_match(match):
            key = match.group(1)
            return d.get(key, match.group(0))
        return x.sub(replace_match, text)
    # end of debracker

    
# end of class MarvinMcMarvelous

if __name__ == "__main__":
    MarvinMcMarvelous().main()

# EOF
#############################################################################
