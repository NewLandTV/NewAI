const { Client, Events, GatewayIntentBits } = require('discord.js');
const { LocalModel, LocalSpeech } = require('./local-model');
const { Conversations } = require('./conversation');

async function main() {
    const token = process.env.DISCORD_BOT_TOKEN?.trim();
    if (!token) throw new Error('로컬 환경에서 DISCORD_BOT_TOKEN을 설정해 주세요.');

    const model = new LocalModel();
    const conversations = new Conversations();
    const speech = process.env.NEWAI_VOICE === '1' ? new LocalSpeech() : null;
    const audio = speech ? new (require('./discord-voice').VoiceSessions)({ speech, model, conversations }) : null;
    const client = new Client({
        intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages,
            ...(audio ? [GatewayIntentBits.GuildVoiceStates] : [])],
        allowedMentions: { parse: [], repliedUser: false },
    });
    const close = () => { audio?.close(); conversations.close(); client.destroy(); model.close(); speech?.close(); };
    process.once('SIGINT', close);
    process.once('SIGTERM', close);

    client.once(Events.ClientReady, () => console.log('NewAI 봇이 준비됐어요. 서버 채널에서 봇을 멘션하고 질문을 보내 주세요.'));
    client.on(Events.Error, () => console.error('디스코드 연결에 문제가 생겼어요.'));
    if (audio) client.on(Events.VoiceStateUpdate, (_, state) => audio.onVoiceState(state, client.user?.id));
    client.on(Events.MessageCreate, async (message) => {
        if (message.author.bot || !message.inGuild() || !message.mentions.users.has(client.user.id)) return;
        const prompt = message.content.replace(new RegExp(`<@!?${client.user.id}>`, 'g'), '').trim();
        if (!prompt) return;

        const contextKey = `text:${message.guildId}:${message.channelId}:${message.author.id}`;
        let response;
        try {
            if (prompt === '대화 초기화') {
                conversations.clear(contextKey);
                response = '이 채널에서 저와 나눈 최근 대화 맥락을 지웠어요.';
            } else if (['음성 시작', '음성 시작 인사', '음성 종료'].includes(prompt)) {
                response = !audio ? '이 봇은 아직 음성 대화가 꺼져 있어요.' : prompt === '음성 종료'
                    ? audio.leave(message) : await audio.join(message, prompt === '음성 시작 인사');
            } else {
                response = await conversations.ask(model, contextKey, prompt) ||
                    '답변을 만들지 못했어요. 가장 궁금한 한 가지를 짧게 말해 줄래요?';
            }
        } catch (error) {
            response = error.code === 'BUSY'
                ? '다른 질문에 답하고 있어요. 잠시 후 다시 시도해 주세요.'
                : error.code === 'TIMEOUT' ? '답변을 만드는 데 시간이 너무 오래 걸렸어요. 궁금한 점을 한 가지씩 물어봐 주세요.'
                    : '로컬 모델로 답변을 만들지 못했어요. 문제가 계속되면 관리자에게 알려 주세요.';
        }
        try {
            await message.reply(response);
        } catch {
            console.error('디스코드에서 답변을 전송하지 못했어요.');
        }
    });

    try {
        await Promise.all([model.ready, speech?.ready]);
        await client.login(token);
    } catch {
        close();
        throw new Error('시작하지 못했어요. 선택한 로컬 모델, 음성 설정과 디스코드 봇 토큰을 확인해 주세요.');
    }
}

main().catch((error) => { console.error(error.message); process.exitCode = 1; });
