const { Client, Events, GatewayIntentBits } = require('discord.js');
const { LocalModel } = require('./local-model');

async function main() {
    const token = process.env.DISCORD_BOT_TOKEN?.trim();
    if (!token) throw new Error('로컬 환경에서 DISCORD_BOT_TOKEN을 설정해 주세요.');

    const model = new LocalModel();
    const client = new Client({
        intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages],
        allowedMentions: { parse: [], repliedUser: false },
    });
    const close = () => { client.destroy(); model.close(); };
    process.once('SIGINT', close);
    process.once('SIGTERM', close);

    client.once(Events.ClientReady, () => console.log('NewAI 봇이 준비됐어요. 서버 채널에서 봇을 멘션하고 질문을 보내 주세요.'));
    client.on(Events.Error, () => console.error('디스코드 연결에 문제가 생겼어요.'));
    client.on(Events.MessageCreate, async (message) => {
        if (message.author.bot || !message.inGuild() || !message.mentions.users.has(client.user.id)) return;
        const prompt = message.content.replace(new RegExp(`<@!?${client.user.id}>`, 'g'), '').trim();
        if (!prompt) return;

        let response;
        try {
            response = await model.ask(prompt) || '답변을 만들지 못했어요. 질문을 바꿔 다시 시도해 주세요.';
        } catch (error) {
            response = error.code === 'BUSY'
                ? '다른 질문에 답하고 있어요. 잠시 후 다시 시도해 주세요.'
                : '로컬 모델로 답변을 만들지 못했어요. 문제가 계속되면 관리자에게 알려 주세요.';
        }
        try {
            await message.reply(response);
        } catch {
            console.error('디스코드에서 답변을 전송하지 못했어요.');
        }
    });

    try {
        await model.ready;
        await client.login(token);
    } catch {
        close();
        throw new Error('시작하지 못했어요. 로컬 모델 설정과 디스코드 봇 토큰을 확인해 주세요.');
    }
}

main().catch((error) => { console.error(error.message); process.exitCode = 1; });
