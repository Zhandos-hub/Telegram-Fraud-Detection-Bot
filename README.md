# I created this TG bot for education purpuses (experimenting with voice recognition tools and exploring TG bot library) . Nothing new but still may be useful for someone.

This is a small side project that listens to voice messages on Telegram and tries to guess if the person is trying to scam you.
Nothing fancy, just some Python, speech recognition, and a bit of AI magic glued together.

Now it works for russian language (to add bit of uniqness) but can be easily changed by modifing FRAUD_KEYPHRASES list

---

## What it actually does

1. You send a voice or audio file (.ogg, .mp3, etc.) to the bot.
2. It turns it into text using Google Speech Recognition.
3. Then it scans that text for sketchy stuff like:

   * "служба безопасности банка"
   * "код из смс"
   * anything that sounds like a money transfer
4. It also checks tone using a small Russian BERT model to see if it sounds pushy, emotional, or shady.
5. Based on all that, it gives you a “risk level” — from chill to “bro, that’s 100% a scam.”

---

## Stack

* Python 3.10+
* `python-telegram-bot` for handling messages
* `SpeechRecognition` + `pydub` for voice to text
* `transformers` (Hugging Face) for text analysis
* `RapidFuzz` for approximate matches
* `dotenv` for keeping secrets out of your code

---

## Setup

Clone it:

```bash
git clone https://github.com/yourusername/fraud-detection-bot.git
cd fraud-detection-bot
```

Create a `.env` file:

```
BOT_TOKEN=your_telegram_bot_token_here
```

(yeah, don’t hardcode your token into the script — `.env` is safer)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run it:

```bash
python main.py
```

That’s it. The bot should go live and start listening for messages.

---

## Notes

* Don’t use it for real security or financial stuff — this is just a side experiment.
* If you push it to GitHub, make sure `.env` is in your `.gitignore`.
* The model isn’t perfect, but it catches most obvious scam talk.

---

## License

MIT — do whatever you want with it. Just don’t sell it as “real fraud detection software.”

---

~ designed & coded by Zhandos (ofc not without help of copilot)
