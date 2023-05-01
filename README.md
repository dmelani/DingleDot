# DingleDot - a stable diffusion bot of questionable quality

Epically crappy discord bot, but it works.

## Features
- Support for Txt2Img to generate images.
- Command to remove last image if needed.
- Interrogate last generated pictures.
- Support for multiple models with default values for VAE.
- Support for multiple samplers.
- Support for filtering NSFW images. This needs a [patched version](https://github.com/dmelani/stable-diffusion-webui-nsfw-censor) of the nsfw filter extension if you want to turn it off and on via the bot. I will remove this requirement if the patched version is merged upstream.
- Support for listing loras. This needs the [sd-api-lora](https://github.com/dmelani/sd-api-lora) extension.
- Support for prompt matrix.

## Prerequisites
- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) installed and started with api enabled. This does not have to be on the same computer as the bot.
- A Discord bot application and a bot token.

## Installation
- Install the python packages listed in requirements.txt.
- Copy dot\_env.example to .env and fill in the values. You need to have a bot token, and you also need to enter the guild ID's of the servers you wish the bot to respond to. 
- Copy render.yaml.example to render.yaml and fill in as needed.
- Invite the bot to your server. It needs to be able to read messages, send messages, manage messages, and attach files.

## Running
- Start with python dingledot.py

## Stuff missing
- Error handling!
