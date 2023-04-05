import discord
import json
from discord.ext import commands
from discord import option
from discord import Option
from discord.ext.commands.errors import CheckFailure
from PIL import Image
import requests
import base64
from io import BytesIO
from argparse import ArgumentParser, ArgumentError

models_LUT = {
        "aom3a2": ("more_models_anime_OrangeMixs_Models_AbyssOrangeMix3_AOM3A2_orangemixs", "orangemix.vae.pt"),
        "deliberate": ("more_models_allround_Deliberate_deliberate_v2", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "chilloutmix": ("more_models_allround_ChilloutMix_chilloutmix_NiPrunedFp32Fix", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "rpg": ("more_models_allround_RPG_rpg_V4", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "rev": ("more_models_allround_Realistic Vision_realisticVisionV20_v20", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "rev_animated": ("more_models_allround_ReV Animated_revAnimated_v11", "kl-f8-anime2.ckpt"),
        "anythingv5": ("more_models_anime_Anything V5_AnythingV3V5_v5PrtRE", "kl-f8-anime2.ckpt")
}

allowed_guilds = None
disallowed_channels = ["general", "allmänt"]

def check_if_allowed_guilds(ctx):
    # Direct messages do not have guild
    if ctx.guild is None:
        return True

    return ctx.guild and ctx.guild.id in allowed_guilds

def check_if_allowed_channels(ctx):
    if not ctx.message:
        return False

    if type(ctx.message.channel) is discord.DMChannel:
        return True

    if ctx.message.channel.name in disallowed_channels:
        return False
    
    return True

class ArgParseException(Exception):
    pass

class NonExitingArgumentParser(ArgumentParser):
    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.

        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        args = {'prog': self.prog, 'message': message}
        raise ArgParseException('%(prog)s: error: %(message)s\n' % args)

class Txt2Img:
    def __init__(self, prompt = "Dingle dot the test bot", negative_prompt = "", steps = 28, sampler="DPM++ SDE Karras", filter_nsfw = True, batch_size=1, model=None, vae=None, width=512, height=512):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.sampler_index = sampler
        self.steps = steps
        self.n_iter = batch_size
        self.width = width
        self.height = height
        
        self.override_settings = {
                "filter_nsfw" : filter_nsfw
        }
        if model:
            self.override_settings["sd_model_checkpoint"] = model
        if vae:
            self.override_settings["sd_vae"] = vae

        self.override_settings_restore_afterwards = True

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)
        
class Txt2ImgResponse:
    def __init__(self, images, parameters, info):
        self.images = images 
        self.parameters = parameters
        self.info = info

def parse_txt2img_respones(data):
    d = json.loads(data)
    return Txt2ImgResponse(d['images'], d['parameters'], d['info'])

pics_args_parse = NonExitingArgumentParser(prog="!render", description="dingledong", add_help=False, exit_on_error=False)
pics_args_parse.add_argument("--nsfw", help="Allow nsfw content", default=False, action='store_true')
pics_args_parse.add_argument("-n", help="Number of pictures", default=1, type=int)
pics_args_parse.add_argument("-m", "--model", dest="data_model", help=f"Stable diffusion model", choices=models_LUT.keys(), default=None, type=str)
pics_args_parse.add_argument("-l", "--layout", dest="layout", default="square", choices=["square", "portrait", "landscape"])
pics_args_parse.add_argument("prompt", type=str)
pics_args_parse.add_argument("neg_prompt", metavar="negative prompt", type=str, nargs='?', default="(bad quality, worst quality:1.4), child, kid, toddler")

dimensions_LUT = {
        "square": (512, 512),
        "landscape": (768, 512),
        "portrait": (512, 768)
        }

class Pics(commands.Cog):
    
    def __init__(self, bot):
        self.bot = bot

    @commands.command(usage=pics_args_parse.format_help())
    @commands.check(check_if_allowed_guilds)
    @commands.check(check_if_allowed_channels)
    async def render(self, ctx, *msg):
        member = ctx.author

        try:
            args = pics_args_parse.parse_args(msg)
        except ArgumentError as e:
            await ctx.send(f"Oi, {member}. Bad command: {e}")
            return

        prompt = args.prompt
        neg_prompt = args.neg_prompt
        batch_size = args.n
        filter_nsfw = False if args.nsfw is True else True
        data_model = args.data_model
        width, height = dimensions_LUT[args.layout]

        if filter_nsfw and "nsfw" not in neg_prompt:
            neg_prompt = "((nsfw)), " + neg_prompt

        if data_model is not None and data_model not in models_LUT:
            await ctx.send(f"Oi, {member}. No such model.")
            return
        
        model = None
        vae = None
        if data_model:
            model, vae = models_LUT[data_model]
        
        await ctx.send(f"Ok, {member}. Rendering {prompt}")

        t = Txt2Img(prompt=prompt, negative_prompt=neg_prompt, filter_nsfw=filter_nsfw, batch_size=batch_size, model=model, vae=vae, width=width, height=height)
        r_data = requests.post('http://192.168.1.43:7860/sdapi/v1/txt2img', data=t.to_json(), headers={'Content-type': 'application/json'})
        #turn this into async
        resp = parse_txt2img_respones(r_data.text)
        files = []
        try:
            for x in resp.images:
                pic = base64.b64decode(x)

                img = Image.open(BytesIO(pic))
                if not img.getbbox():
                    # All black image
                    continue

                f = discord.File(BytesIO(pic), filename="pic.png")
                files.append(f)
        except Exception as e:
            await ctx.send(f"Failed to generate pic, {member}")
            return

        diff_len = batch_size - len(files)
        if diff_len > 0:
            await ctx.send(f"Some pics were too spicy for me")

        if len(files):
            await ctx.send(f"Here you go, {member}", files=files)

    @render.error
    async def render_error(self, ctx, error):
        if type(error) is CheckFailure:
            return

        if type(error.original) is ArgParseException:
            await ctx.send(str(error.original))
        else:
            await ctx.send(f"Well that didn't work... {type(error)}")

def setup(bot):
    global allowed_guilds
    allowed_guilds = [int(x) for x in bot.allowed_guilds]
    bot.add_cog(Pics(bot))
