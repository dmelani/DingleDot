import discord
from discord.ext import commands
from dotenv import dotenv_values
import os

owner = None
config = None

def check_if_owner(ctx):
    return str(ctx.author) == owner

class DingleDot(commands.Bot):
    async def on_ready(self):
        print(f"Logged in as user {self.user}")

    @commands.check(check_if_owner)
    async def reload(self, ctx):
        await ctx.send(f"OK, reloading.")
        for f in os.listdir("./cogs"):
            if f.endswith(".py"):
                self.reload_extension("cogs." + f[:-3])
        await ctx.send(f"Reloading complete.")


async def main():
    intents = discord.Intents.default()
    intents.message_content = True

    dd = DingleDot(intents=intents, command_prefix='!')
    dd.allowed_guilds = config["GUILDS"].split(",")
    dd.allowed_guilds_restricted = config["GUILDS_RESTRICTED"].split(",")
    dd.add_command(commands.Command(dd.reload))

    for f in os.listdir("./cogs"):
        if f.endswith(".py"):
            print(f"Loading cog: {f}")
            dd.load_extension("cogs." + f[:-3])
    print("Cogs loaded")
    
    await dd.start(config["BOT_TOKEN"])
    


if __name__=="__main__":
    import asyncio
    config = dotenv_values(".env")
    owner = config["OWNER"]
    asyncio.run(main())
