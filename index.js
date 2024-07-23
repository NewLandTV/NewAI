// Require
const { Client, GatewayIntentBits } = require("discord.js");
const { Configuration, OpenAIApi } = require("openai");
const { token, apiKey } = require("./config.json");

// Constants
const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent] });
const openAI = new OpenAIApi(new Configuration({ apiKey: apiKey }));
const messages = [];

// Client events
client.once("ready", () => {
    client.user.setActivity("New AI", { type: "PLAYING" });

    console.log(`Logged in as ${client.user.tag}!`);
});

client.on("messageCreate", async message => {
    if (message.author.bot) return;

    messages.push({ role: "user", content: message.content });
    console.log(`${message.member.displayName} : ${message.content}`);

    const completion = await openAI.createChatCompletion({ model: "gpt-3.5-turbo", messages: messages });

    messages.push({ role: "assistant", content: completion.data.choices[0].message.content });
    console.log(`New AI : ${completion.data.choices[0].message.content}`);

    await message.channel.send(completion.data.choices[0].message.content);
});

client.login(token);