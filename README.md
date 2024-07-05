# Marvin-McMarvelous
Marvin McMarvelous, the robotic jive-bot does the follow:
<img src="mcm.png" style="float: right;" align="right" width="512" height="512"/>

1. listen for the wake word (default: "marvin")
2. convert text to speech using whisper
3. sent a request to an LLM
4. send the LLM response to a text-to-speech bot to read it aloud
5. send the LLM response to a text-to-image bot and display the result

The main listen magic was ripped directly from https://huggingface.co/learn/audio-course/chapter7/voice-assistant

## Configuration

### Wake Words

The wake words in MIT/ast-finetuned-speech-commands-v2 are:

```
backward follow five    bed   zero  on     learn
two      house  tree    dog   stop  seven  eight
down     six    forward cat   right visual four
wow      no     nine    off   three left   marvin
yes      up     sheila  happy bird  go     one
```

I stuck with "Marvin" but "Sheila" could also be a good pick. 
If you want to go with "on" or "two" that's your call :-P

### Example uses

#### A silly friend

```
./MarvinMcMarvelous.py --llm_json='{"prompt": "<<prompt>>","model": "phi3","stream": false,"system": "You are a wild card lunatic politically incorrect wacko-bot! Your output is spoken aloud so you try to keep your answers short and sweet and hilarious","options": {"seed": 101,"temperature": 0}}'
```

#### Image Assistant

```
./MarvinMcMarvelous.py --quiet --llm_json='{"prompt": "<<prompt>>","model": "phi3","stream": false,"system": "You are concept artist who describes cool cyberpunk images with an emphasize on female net runners with vr headsets. Your output is read aloud so you keep your responses brief, but it is also used by stable diffusion to generate images so it is also evocative. You always include enough information so that the requested scene is generated","options": {"seed": 101,"temperature": 0}}' --chop
```

Note the ```--chop``` will use the full LLM output for image generation but will only read the first sentence aloud.

# Setup 

I recommend pyenv https://github.com/pyenv/pyenv ; with python >= 3.10.10

```
sudo apt install tk
pyenv virtualenv 3.10.10 marvin_mcmarvelous
pyenv activate marvin_mcmarvelous
pyenv local marvin_mcmarvelous
pip install -r requirements.txt 
./MarvinMcMarvelous.py
```

It's likely you will need a huggingface account and token set up 

## Local Bots Needed

The convention I'm using is to setup a host entry for "aid" to point to the AI host.
At the moment, the LLM, TTS and TTI are all accessed over REST

### LLM: Ollama: https://ollama.com/

By default it should run on http://aid:11434/api/generate 

I usually run it like so:

```
sudo systemctl stop ollama.service
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

I like to use phi3 but it can be a little overly sensitive. YMMV.

### TTS: Piper : https://github.com/rhasspy/piper

By default it should run on http://aid:5000/

For now use my branch: https://github.com/luckybit4755/piper/tree/http-server-json-response/ to get patch to handle JSOn request / responses and chop text into sentences with NLTK.

Voice preview here: https://rhasspy.github.io/piper-samples/

### TTI: Automatic1111

https://github.com/AUTOMATIC1111/stable-diffusion-webui/
		
By default it shoudl run on http://aid:7860/sdapi/v1/txt2img

I'm not going to go into this a lot because the dox for it are already super great, but recommend running it with: ```./webui.sh --xformers --api --listen```

# More Marvelousnessitudinesses!

If you are more focusing on image generation you can use ```--chop``` to have Marvin only read the first sentence while still using the full llm output for the image prompt.

You can make Marvin be a little quieter at startup and shutdown with the ```--quiet``` flag.

## System prompts

You can override the systems prompts a few ways:

1. using the ```--system="System prompt goes where"```
2. using the ```--load=personality.json```
3. dynamially using ```--wake_words=marvin,learn```, saying "learn" will let you speak a new prompt

You can also use a longer prompt defined in ```SystemPrompts.py``` like ```--system=dan```
