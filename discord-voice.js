const { Readable } = require('node:stream');
const { performance } = require('node:perf_hooks');
const OpusScript = require('opusscript');
const { ChannelType } = require('discord.js');
const voice = require('@discordjs/voice');

const MAX_AUDIO = 48_000 * 4 * 12;
const SILENCE_MS = 650;
const IDLE_MS = 5 * 60_000;
const GREETING = '안녕하세요, 뉴 에이아이예요. 지금부터 대화할 수 있어요. 어떤 이야기부터 할까요?';

function speechText(input) {
    const text = input.replace(/https?:\/\/\S+/gi, '').replace(/<[@#][^>]+>/g, '').replace(/[`*_#~]/g, '')
        .replace(/\s+/g, ' ').trim();
    const chars = Array.from(text);
    if (chars.length <= 180) return text;
    const clipped = chars.slice(0, 180).join('');
    const lastSentence = [...clipped.matchAll(/[.!?。！？](?=\s|$)/gu)].at(-1);
    if (lastSentence && lastSentence.index > clipped.length / 2) return clipped.slice(0, lastSentence.index + 1);
    const lastWord = clipped.lastIndexOf(' ');
    return lastWord > clipped.length / 2 ? `${clipped.slice(0, lastWord)}…` : `${chars.slice(0, 179).join('')}…`;
}

function canRecord(session, speakerId) {
    return !session.closed && !session.busy && !session.capture && speakerId === session.owner;
}

function captureOpus(stream, onAudio) {
    const decoder = new OpusScript(48_000, 2, OpusScript.Application.VOIP);
    let buffers = [], size = 0, done = false;
    const finish = (deliver) => {
        if (done) return;
        done = true;
        clearTimeout(timeout);
        decoder.delete();
        stream.destroy();
        const pcm = deliver && size >= 48_000 * 4 * .28 ? Buffer.concat(buffers, size) : null;
        buffers = [];
        onAudio(pcm);
    };
    const timeout = setTimeout(() => finish(true), 12_000);
    stream.on('data', packet => {
        if (done) return;
        try {
            const chunk = decoder.decode(packet).subarray(0, MAX_AUDIO - size);
            size += chunk.length;
            buffers.push(chunk);
            if (size >= MAX_AUDIO) finish(true);
        } catch {
            finish(false);
        }
    });
    stream.once('end', () => finish(true));
    stream.once('close', () => finish(false));
    stream.once('error', () => finish(false));
    return () => finish(false);
}

class VoiceSessions {
    constructor({ speech, model, conversations }) {
        this.speech = speech;
        this.model = model;
        this.conversations = conversations;
        this.session = null;
    }

    touch(session) {
        clearTimeout(session.idle);
        session.idle = setTimeout(() => this.close(session), IDLE_MS);
        session.idle.unref();
    }

    async join(message, greet) {
        if (this.session) return '이미 음성 대화가 진행 중이에요. 시작한 사람이 음성 종료를 보내면 다시 시작할 수 있어요.';
        if (this.speech.closed) return '로컬 음성 처리를 사용할 수 없어요. 봇을 다시 실행해 주세요.';
        const channel = message.member?.voice?.channel;
        if (channel?.type !== ChannelType.GuildVoice) return '먼저 서버의 일반 음성 채널에 들어가 주세요.';
        if (!channel.joinable || !channel.speakable) return '이 음성 채널에 들어가거나 말할 권한이 없어요.';
        const connection = voice.joinVoiceChannel({ guildId: message.guildId, channelId: channel.id,
            adapterCreator: message.guild.voiceAdapterCreator, selfMute: false, selfDeaf: false });
        const session = { owner: message.author.id, guild: message.guildId, channel: channel.id,
            key: Symbol('voice-session'), connection, player: voice.createAudioPlayer(),
            closed: false, busy: true, capture: null, abort: new AbortController() };
        this.session = session;
        connection.on('error', () => this.close(session));
        connection.on('stateChange', (_, current) => {
            if ([voice.VoiceConnectionStatus.Destroyed, voice.VoiceConnectionStatus.Disconnected].includes(current.status)) {
                this.close(session);
            }
        });
        session.player.on('error', () => console.warn('[voice] 음성 재생 실패'));
        try {
            await voice.entersState(connection, voice.VoiceConnectionStatus.Ready, 12_000);
            if (session.closed) return '음성 대화가 종료됐어요.';
            connection.subscribe(session.player);
            session.onSpeaking = speaker => this.onSpeaking(session, speaker);
            connection.receiver.speaking.on('start', session.onSpeaking);
            this.touch(session);
            if (greet) {
                this.speak(session, GREETING).catch(() => console.warn('[voice] 첫 인사 재생 실패'))
                    .finally(() => { session.busy = false; });
            } else {
                session.busy = false;
            }
            return '음성 대화를 시작했어요. 시작한 사람의 음성만 로컬에서 인식하고, 녹음은 저장하지 않아요. 답변은 채널에서 들리며 AI 음성이에요. 멘션 후 `음성 종료`로 나갈 수 있어요.';
        } catch {
            this.close(session);
            return '음성 채널에 연결하지 못했어요. 채널 권한과 네트워크를 확인해 주세요.';
        }
    }

    onSpeaking(session, userId) {
        if (!canRecord(session, userId)) return;
        this.touch(session);
        const stream = session.connection.receiver.subscribe(userId, {
            end: { behavior: voice.EndBehaviorType.AfterSilence, duration: SILENCE_MS },
        });
        session.capture = captureOpus(stream, pcm => {
            session.capture = null;
            if (!pcm || session.closed) return;
            session.busy = true;
            this.reply(session, pcm).catch(() => console.warn('[voice] 로컬 음성 처리 실패'))
                .finally(() => { session.busy = false; });
        });
    }

    async reply(session, pcm) {
        const start = performance.now();
        const transcript = await this.speech.transcript(pcm);
        const sttMs = performance.now() - start;
        if (!transcript || session.closed) return;
        let answer;
        try {
            answer = await this.conversations.ask(this.model, session.key, transcript, true);
        } catch (error) {
            if (error.code === 'BUSY') answer = '지금 다른 질문에 답하고 있어요. 잠시 후 다시 말해 주세요.';
            else if (error.code === 'TIMEOUT') answer = '답변을 만드는 데 시간이 오래 걸렸어요. 한 가지씩 짧게 물어봐 주세요.';
            else throw error;
        }
        const llmMs = performance.now() - start - sttMs;
        if (session.closed) return;
        const ttsStarted = performance.now();
        await this.speak(session, answer || '질문을 듣긴 했지만 답을 만들지 못했어요. 짧게 다시 말해 줄래요?', () => {
            console.info(`[voice] stt=${Math.round(sttMs)}ms llm=${Math.round(llmMs)}ms tts_to_play=${Math.round(performance.now() - ttsStarted)}ms after_capture_to_play=${Math.round(performance.now() - start)}ms`);
        });
    }

    async speak(session, text, onPlay = () => {}) {
        const phrase = speechText(text);
        if (session.closed || !phrase) return;
        // Synthesize a short reply before playback: on slower CPUs a later chunk
        // would not arrive before the voice player runs out of Opus frames.
        const audio = await this.speech.synth(phrase);
        if (!audio.length || session.closed) return;
        const pcm = Readable.from([audio], { objectMode: false });
        const resource = voice.createAudioResource(pcm, { inputType: voice.StreamType.Raw });
        session.player.play(resource);
        try {
            await voice.entersState(session.player, voice.AudioPlayerStatus.Playing, AbortSignal.any([
                session.abort.signal, AbortSignal.timeout(5000),
            ]));
            onPlay();
            await voice.entersState(session.player, voice.AudioPlayerStatus.Idle, AbortSignal.any([
                session.abort.signal, AbortSignal.timeout(45_000),
            ]));
        } finally {
            pcm.destroy();
            session.player.stop(true);
        }
    }

    leave(message) {
        const session = this.session;
        if (!session || session.guild !== message.guildId) return '이 서버에서 진행 중인 음성 대화가 없어요.';
        if (session.owner !== message.author.id) return '음성 대화를 시작한 사람이 종료할 수 있어요.';
        this.close(session);
        return '음성 대화를 종료했어요.';
    }

    onVoiceState(state, botId) {
        const session = this.session;
        if (session && state.guild.id === session.guild && state.channelId !== session.channel &&
            (state.id === session.owner || state.id === botId)) this.close(session);
    }

    close(session = this.session) {
        if (!session || session.closed) return;
        session.closed = true;
        session.abort.abort();
        clearTimeout(session.idle);
        session.capture?.();
        if (session.onSpeaking) session.connection.receiver.speaking.off('start', session.onSpeaking);
        session.player.stop(true);
        if (session.connection.state.status !== voice.VoiceConnectionStatus.Destroyed) session.connection.destroy();
        this.conversations.clear(session.key);
        if (this.session === session) this.session = null;
    }
}

module.exports = { VoiceSessions, speechText, canRecord, captureOpus };
