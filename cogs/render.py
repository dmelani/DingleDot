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
import argunparse

import aiohttp

models_LUT = {
        "anythingv5": ("more_models_anime_Anything V5_AnythingV3V5_v5PrtRE", "kl-f8-anime2.ckpt"),
        "aom3a1b": ("more_models_anime_OrangeMixs_Models_AbyssOrangeMix3_AOM3A1B_orangemixs", "orangemix.vae.pt"),
        "aom3a1": ("more_models_anime_OrangeMixs_Models_AbyssOrangeMix3_AOM3A1_orangemixs", "orangemix.vae.pt"),
        "aom3a2": ("more_models_anime_OrangeMixs_Models_AbyssOrangeMix3_AOM3A2_orangemixs", "orangemix.vae.pt"),
        "aom3a3": ("more_models_anime_OrangeMixs_Models_AbyssOrangeMix3_AOM3A3_orangemixs", "orangemix.vae.pt"),
        "aom3": ("more_models_anime_OrangeMixs_Models_AbyssOrangeMix3_AOM3_orangemixs", "orangemix.vae.pt"),
        "aurora": ("more_models_anime_Aurora_AuroraONE", "kl-f8-anime2.ckpt"),
        "chilloutmix": ("more_models_allround_ChilloutMix_chilloutmix_NiPrunedFp32Fix", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "counterfeit": ("more_models_anime_Counterfeit-V2.5_Counterfeit-V2.5", "Counterfeit-V2.5.vae.pt"),
        "darksushi": ("more_models_anime_DarkSushiMix_darkSushiMixMix_brighterPruned", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "darksushi25d": ("more_models_anime_darksushi2.5d_darkSushi25D25D_v20", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "deliberate": ("more_models_allround_Deliberate_deliberate_v2", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "ghostmix": ("more_models_anime_cetus_ghostmix_ghostmix_v11", "kl-f8-anime2.ckpt"),
        "hassanblend": ("more_models_allround_HassanBlend 1.5.1.2_hassanblend1512And_hassanblend1512", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "hassanblendfantasy": ("more_models_allround_HassanBlend-Fantasy_HB-Fantasy1.5", "vae-ft-mse-840000-ema-pruned.safetensors"),
#        "illuminati": ("more_models_allround_Illuminati Diffusion v1.1_illuminatiDiffusionV1_v11", "Automatic"),
        "illuminati": ("more_models_allround_Illuminati Diffusion v1.1_illuminatiDiffusionV1_v11", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "illuminutty": ("more_models_allround_Illuminutty Diffusion_illuminuttyDiffusion_v111", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "kenshi": ("more_models_anime_Kenshi_kenshi_01", "kl-f8-anime2.ckpt"),
        "lyriel": ("more_models_allround_Lyriel_lyriel_v15", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "ned": ("more_models_anime_NeverEnding Dream (NED)_neverendingDreamNED_bakedVae", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "pw": ("more_models_nsfw_Perfect World_perfectWorld_v2Baked", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "realism": ("more_models_allround_Realism Engine_realismEngine_v10", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "rev_animated": ("more_models_allround_ReV Animated_revAnimated_v122", "kl-f8-anime2.ckpt"),
        "rev": ("more_models_allround_Realistic Vision_realisticVisionV20_v20", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "rpg": ("more_models_allround_RPG_rpg_V4", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "simplybeautiful": ("more_models_anime_SimplyBeautiful_simplyBeautiful_v10", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "test5": ("more_models_nsfw_Uber_Realistic_Porn_Merge_(URPM)_uberRealisticPornMerge_urpmv13", "vae-ft-mse-840000-ema-pruned.safetensor"),
        "test10": ("more_models_test10", "kl-f8-anime2.ckpt"),
        "test11": ("more_models_test11", "kl-f8-anime2.ckpt"),
        "test12": ("more_models_test12", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "test_aom3a2_aurora_6040": ("test_aom3a2_aurora_6040", "kl-f8-anime2.ckpt"),
        "test_aom3a2_aurora_8020": ("test_aom3a2_aurora_8020", "orangemix.vae.pt"),
        "test_aom3a2_aurora_9010": ("test_aom3a2_aurora_9010", "orangemix.vae.pt"),
        "mixpro": ("more_models_anime_Mix-Pro_mixProV4_v4", "kl-f8-anime2.ckpt"),
        "citrine": ("more_models_anime_CitrineDreamMix_citrinedreammix_v11BakedVAE", "kl-f8-anime2.ckpt"),
        "v08": ("more_models_testing_allround_V08_V08_V08", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "v08a": ("more_models_testing_allround_V08_V08_V08a", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "v08c": ("more_models_testing_allround_V08_V08_V08c", "vae-ft-mse-840000-ema-pruned.safetensors"),
        "anyh": ("more_models_anime_AnyHentai_anyhentai_20", "Counterfeit-V2.5.vae.pt"),
}

sampler_LUT = {
        "euler_a": ("Euler a", 20),
        "ddim": ("DDIM", 50),
        "dpmpp_sde_ka": ("DPM++ SDE Karras", 31),
        "dpmpp_2m_ka": ("DPM++ 2M Karras", 31),
        "dpmpp_2s_a_ka": ("DPM++ 2S a Karras", 31),
        "heun": ("Heun", 50)
        }

dimensions_LUT = {
        "square": (512, 512),
        "lsquare": (768, 768),
        "landscape": (768, 512),
        "llandscape": (1024, 768),
        "portrait": (512, 768),
        "lportrait": (768, 1024)
        }

upscalers_LUT = {
        "normal": "R-ESRGAN 4x+",
        "anime" : "R-ESRGAN 4x+ Anime6B", 
        "ultrasharp" : "4x-UltraSharp",
        "ultramixbalanced" : "4x-UltraMix_Balanced"
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

class Interrogate:
    def __init__(self, image, model):
        self.image = image
        self.model = model 
        
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)

class InterrogateResponse:
    def __init__(self, data):
        message = json.loads(data)
        self.message = message["caption"]

def parse_interrogate_respones(data):
    return InterrogateResponse(data)

class Txt2Img:
    def __init__(self, prompt = "Dingle dot the test bot", negative_prompt = "", sampler_name="DPM++ SDE Karras", steps=30, filter_nsfw = True, batch_size=1, model=None, vae=None, width=512, height=512, clip_stop=1, restore_faces=False, cfg_scale=7, upscaler=None, seed=None):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.sampler_name = sampler_name
        self.steps = steps
        self.n_iter = batch_size
        self.width = width
        self.height = height
        self.restore_faces = restore_faces
        self.cfg_scale = cfg_scale

        self.override_settings = {
            "filter_nsfw" : filter_nsfw,
            "CLIP_stop_at_last_layers": clip_stop
        }
        if model:
            self.override_settings["sd_model_checkpoint"] = model
        if vae:
            self.override_settings["sd_vae"] = vae

        if seed:
            self.seed = seed

        if upscaler:
            self.hr_upscaler = upscaler
            self.enable_hr = True
            self.hr_scale = 2
            self.denoising_strength = 0.15

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
pics_args_parse.add_argument("--cfgs", help="Classifier Free Guidance Scale - how strongly the image should conform to prompt - lower values produce more creative results. Default is 7.", default=7, type=int)
pics_args_parse.add_argument("-m", "--model", dest="data_model", help="Stable diffusion model. See !models for a list", default=[], type=str, action='append')
pics_args_parse.add_argument("-s", "--sampler", dest="sampler_name", help=f"Stable diffusion sampler", choices=sampler_LUT.keys(), default="dpmpp_sde_ka", type=str)
pics_args_parse.add_argument("-i", dest="sampler_steps", help="Number of sampler steps", default=None, type=int)
pics_args_parse.add_argument("-l", "--layout", dest="layout", default="square", choices=["square", "lsquare", "portrait", "lportrait", "landscape", "llandscape"])
pics_args_parse.add_argument("--clip_stop", dest="clip_stop", help="Sets where to stop the CLIP language model. Works kinda like this in layers person -> male, female -> man, boy, woman girl -> and so on", default=1, choices=range(1, 5), type=int)
pics_args_parse.add_argument("prompt", type=str)
pics_args_parse.add_argument("neg_prompt", metavar="negative prompt", type=str, nargs='?', default="(bad quality, worst quality:1.4)")
pics_args_parse.add_argument("--restore_faces", help="Attempts to restore faces", default=False, action='store_true')
pics_args_parse.add_argument("-U", "--upscale", dest="upscaler", help=f"Upscale by 2x", default=None, choices=upscalers_LUT.keys())
pics_args_parse.add_argument("--seed", help="seed", default=None, type=int)

again_args_parse = NonExitingArgumentParser(prog="!again", add_help=False, exit_on_error=False)
again_args_parse.add_argument("-m", "--model", dest="data_model", help="Stable diffusion model. See !models for a list", default=None, type=str)

explain_args_parse = NonExitingArgumentParser(prog="!explain", add_help=False, exit_on_error=False)
explain_args_parse.add_argument("last_pic", type=int, default=0)
explain_args_parse.add_argument("-m", dest="model", default="clip", choices=["clip", "deepdanbooru"])

class Pics(commands.Cog):
    
    def __init__(self, bot):
        self.bot = bot
        self.history = {}

    def _make_message_key(self, ctx):
        g = None
        if ctx.guild is not None:
            g = ctx.guild.id

        c = None
        if type(ctx.message.channel) is not discord.DMChannel:
            c = ctx.message.channel
    
        u = ctx.message.author

        return (g, c, u)
    @commands.command()
    @commands.check(check_if_allowed_guilds)
    @commands.check(check_if_allowed_channels)
    async def ohshit(self, ctx):
        member = ctx.author

        key = self._make_message_key(ctx)

        v = self.history.get(key)
        if v is None:
            return

        msg, answer, _, _= v
        if answer is None:
            return

        await answer.delete()
        self.history.pop(key)

    @commands.command(usage=explain_args_parse.format_help())
    @commands.check(check_if_allowed_guilds)
    @commands.check(check_if_allowed_channels)
    async def explain(self, ctx, *msg):
        member = ctx.author
        args = explain_args_parse.parse_args(msg)
        pic_no = args.last_pic
        model = args.model
        key = self._make_message_key(ctx)

        old, _, images, _ = self.history.get(key)
        if old is None:
            await ctx.send(f"Oi, {member}. No previous command.")
            return

        if pic_no > len(images):
            await ctx.send(f"Oi, {member}. No such picture.")
            return

        img = images[pic_no]
        t = Interrogate(img, model)
        async with aiohttp.ClientSession() as session:
            async with session.post('http://192.168.1.43:7860/sdapi/v1/interrogate', data=t.to_json(), headers={'Content-type': 'application/json'}) as response:
                r_data = await response.text()

        resp = parse_interrogate_respones(r_data)

        await ctx.send(f"Oi, {member}. Explanation was: {resp.message}")

    @commands.command(usage=again_args_parse.format_help())
    @commands.check(check_if_allowed_guilds)
    @commands.check(check_if_allowed_channels)
    async def again(self, ctx, *msg):
        member = ctx.author

        key = self._make_message_key(ctx)

        v = self.history.get(key)
        if v is None:
            await ctx.send(f"Oi, {member}. No previous command.")
            return

        old_msg, _, _, _ = v
        if old_msg is None:
            await ctx.send(f"Oi, {member}. No previous command.")
            return

        try:
            new_args = again_args_parse.parse_args(msg)
        except ArgumentError as e:
            await ctx.send(f"Oi, {member}. Bad command: {e}")
            return

        if new_args.data_model is not None:
            old_args = pics_args_parse.parse_args(old_msg)
            prompt = old_args.prompt
            neg_prompt = old_args.neg_prompt
            sampler_name = old_args.sampler_name
            sampler_steps = old_args.sampler_steps
            upscaler = old_args.upscaler

            kwargs = vars(old_args)
            kwargs.pop("data_model")
            kwargs.pop("prompt")
            kwargs.pop("neg_prompt")
            kwargs.pop("sampler_name")
            kwargs.pop("sampler_steps")
            kwargs.pop("upscaler")

            kwargs['model'] = new_args.data_model
            kwargs['sampler'] = sampler_name
            kwargs['i'] = sampler_steps
            kwargs['upscale'] = upscaler

            unparser = argunparse.ArgumentUnparser(opt_value=' ') 
            unp = unparser.unparse_options(kwargs, to_list=True)
            unp += [prompt]
            unp += [neg_prompt]

            await self.render(ctx, *unp)
            return

        await self.render(ctx, *old_msg)

    @commands.command()
    @commands.check(check_if_allowed_guilds)
    @commands.check(check_if_allowed_channels)
    async def lastprompt(self, ctx):
        member = ctx.author

        key = self._make_message_key(ctx)

        tmp = self.history.get(key)
        if tmp is None:
            await ctx.send("Oi, {}. No earlier prompt.".format(member))
            return

        _, _, _, prompts = tmp
        await ctx.send("Oi, {}. The last prompt was: {}\nNegative: {}".format(member, prompts[0], prompts[1]))
     
    @commands.command()
    @commands.check(check_if_allowed_guilds)
    @commands.check(check_if_allowed_channels)
    async def models(self, ctx):
        member = ctx.author

        ms = ', '.join(models_LUT.keys())
        msg = f"Available models: {ms}"
        await ctx.send(msg)


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
        
        print(f"RENDER by {member}: {msg}")

        prompt = args.prompt
        neg_prompt = args.neg_prompt
        batch_size = args.n
        filter_nsfw = False if args.nsfw is True else True
        data_models = args.data_model
        width, height = dimensions_LUT[args.layout]
        clip_stop = args.clip_stop
        restore_faces = args.restore_faces
        cfgs = args.cfgs
        sampler = args.sampler_name
        upscaler = args.upscaler
        sampler_steps = args.sampler_steps
        seed = args.seed

        if len(data_models) == 0:
            data_models = ["deliberate"]

        if filter_nsfw and "nsfw" not in neg_prompt:
            neg_prompt = "(nsfw:1.1), " + neg_prompt

        for dm in data_models:
            if dm not in models_LUT:
                await ctx.send(f"Oi, {member}. No such model:", dm)
                return

        # This is probable a good idea
        neg_prompt = "(child), (kid), (toddler), " + neg_prompt
#        print("DEBUG: final neg prompt", neg_prompt)

        sampler_name = None
        steps = None
        if sampler:
            sampler_name, steps = sampler_LUT[sampler]

        if sampler_steps:
            steps = sampler_steps
    
        upscaler_name = None
        if upscaler:
            upscaler_name = upscalers_LUT[upscaler]
        
        await ctx.send(f"Ok, {member}. GPU goes brrrr!")

        files = []
        used_seeds = []
        images = []
        for dm in data_models:
            model, vae = models_LUT[dm]
            t = Txt2Img(prompt=prompt, negative_prompt=neg_prompt, filter_nsfw=filter_nsfw, batch_size=batch_size, model=model, vae=vae, width=width, height=height, clip_stop=clip_stop, restore_faces=restore_faces, cfg_scale=cfgs, sampler_name=sampler_name, steps=steps, upscaler=upscaler_name, seed=seed)
            async with aiohttp.ClientSession() as session:
                async with session.post('http://192.168.1.43:7860/sdapi/v1/txt2img', data=t.to_json(), headers={'Content-type': 'application/json'}) as response:
                    r_data = await response.text()

            resp = parse_txt2img_respones(r_data)
            #print("DEBUG:", resp.info)#, resp.parameters)
            info = json.loads(resp.info)

            try:
                for i, x in enumerate(resp.images):
                    pic = base64.b64decode(x)

                    img = Image.open(BytesIO(pic))
                    if not img.getbbox():
                        # All black image
                        continue

                    images.append(x)
                    f = discord.File(BytesIO(pic), filename="pic.png")
                    files.append(f)
                    curr_seed = int(info['all_seeds'][i])
                    used_seeds.append(curr_seed)

            except Exception as e:
                await ctx.send(f"Failed to generate pic, {member}")
                return

        used_seeds = [str(x) for x in used_seeds]
        diff_len = (batch_size * len(data_models)) - len(files)
        if diff_len > 0:
            await ctx.send(f"Some pics were too spicy for me")

        answer = None
        if len(files):
            answer = await ctx.send(f"Here you go, {member}. Seeds: {', '.join(used_seeds)}", files=files)

        key = self._make_message_key(ctx)
        self.history[key] = (msg, answer, images, (prompt, neg_prompt))

    @render.error
    async def render_error(self, ctx, error):
        if type(error) is CheckFailure:
            return

        if type(error.original) is ArgParseException:
            await ctx.send(str(error.original))
        else:
            print("DEBUG", error)
            await ctx.send(f"Well that didn't work... {type(error)}")

def setup(bot):
    global allowed_guilds
    allowed_guilds = [int(x) for x in bot.allowed_guilds]
    bot.add_cog(Pics(bot))
