# New AI

Chat bot + OpenAI gpt-3.5-turbo.

## How to run

### Clone this repository

```sh
git clone https://github.com/NewLandTV/NewAI.git
cd NewAI
```

### Install node packages

```sh
npm install
```

### Setup config.json

You should create a config.json file on root of this repository. Then, please write your bot's token and your OpenAI apiKey as shown below...

```json
{
    "token": "[your-discord-bot-token]",
    "apiKey": "[your-open-ai-api-key]"
}
```

### Run bot

Run the run.bat file or follow below.

```sh
node .  # node index.js
```