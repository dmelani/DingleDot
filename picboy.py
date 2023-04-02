import discord
from discord.ext import commands
from dotenv import dotenv_values
import os

config = None

class PicBoy(commands.Bot):
    async def on_ready(self):
        print(f"Logged in as user {self.user}")

    async def reload(self, ctx):
        await ctx.send(f"OK, reloading.")
        for f in os.listdir("./cogs"):
            if f.endswith(".py"):
                self.reload_extension("cogs." + f[:-3])
        await ctx.send(f"Reloading complete.")


async def main():
    intents = discord.Intents.default()
    intents.message_content = True

    pb = PicBoy(intents=intents, command_prefix='!')
    pb.add_command(commands.Command(pb.reload))

    for f in os.listdir("./cogs"):
        if f.endswith(".py"):
            print(f"Loading cog: {f}")
            pb.load_extension("cogs." + f[:-3])
    print("Cogs loaded")
    
    await pb.start(config["BOT_TOKEN"])
    


if __name__=="__main__":
    import asyncio
    config = dotenv_values(".env")
    asyncio.run(main())
