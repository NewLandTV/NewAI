const TTL = 15 * 60_000;

class Conversations {
    constructor({ now = Date.now, ttl = TTL } = {}) {
        this.now = now;
        this.ttl = ttl;
        this.entries = new Map();
        this.sweep = setInterval(() => this.expire(), Math.min(this.ttl, 60_000));
        this.sweep.unref();
    }

    expire() {
        const now = this.now();
        for (const [key, entry] of this.entries) {
            if (entry.expires <= now) this.entries.delete(key);
        }
    }

    open(key) {
        this.expire();
        let entry = this.entries.get(key);
        if (!entry) {
            if (this.entries.size >= 200) this.entries.delete(this.entries.keys().next().value);
            entry = { turns: [], expires: this.now() + this.ttl };
            this.entries.set(key, entry);
        }
        return entry;
    }

    clear(key) { this.entries.delete(key); }

    async ask(model, key, prompt, voice = false) {
        if (!['context', 'research', 'ua'].includes(model.engine)) return model.ask(prompt, { voice });
        prompt = prompt.slice(0, 800);
        const entry = this.open(key);
        const response = await model.ask(prompt, { history: entry.turns.slice(), voice });
        if (response && this.entries.get(key) === entry && entry.expires > this.now()) {
            entry.turns = [...entry.turns, { user: prompt, assistant: response.slice(0, 800) }].slice(-3);
            entry.expires = this.now() + this.ttl;
        }
        return response;
    }

    close() { clearInterval(this.sweep); this.entries.clear(); }
}

module.exports = { Conversations };
