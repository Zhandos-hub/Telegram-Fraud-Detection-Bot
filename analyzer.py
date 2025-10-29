import sys
import os
import subprocess
import re
import logging
import io
import tempfile
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from html import escape

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настраиваем базовое логирование
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Установка и проверка зависимостей ---

# (Имя для pip, Имя для import)
PACKAGE_MAPPING: List[Tuple[str, str]] = [
    ("speech_recognition", "speech_recognition"),
    ("pydub", "pydub"),
    ("rapidfuzz", "rapidfuzz"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("python-telegram-bot", "telegram"),
    ("python-dotenv", "dotenv"),
]

def install_and_import(package_list: List[Tuple[str, str]]):
    logging.info("Checking and installing required packages...")
    for pip_name, import_name in package_list:
        try:
            __import__(import_name)
        except ImportError:
            logging.warning(f"Package {import_name} not found. Installing {pip_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "-q", pip_name])
                logging.info(f"Successfully installed: {pip_name}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to install {pip_name}: {e}")
                sys.exit(1)
        try:
            __import__(import_name)
        except Exception as e:
            logging.error(f"Failed to import {import_name} after install: {e}")
            sys.exit(1)

install_and_import(PACKAGE_MAPPING)

import speech_recognition as sr
from pydub import AudioSegment
from rapidfuzz import fuzz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


BOT_TOKEN = os.getenv("BOT_TOKEN") or "YOUR_BOT_TOKEN_HERE"
MODEL_NAME = "cointegrated/rubert-tiny2-cedr-emotion-detection"
tokenizer = None
model = None


def load_ai_model():
    """Загружает токенизатор и модель Hugging Face в глобальные переменные."""
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        logging.info("AI sentiment model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load AI model. Sentiment analysis will be unavailable. Reason: {e}")
        tokenizer = None
        model = None

load_ai_model()

#settings
FUZZY_THRESHOLD = 70  
MAX_TRANSCRIPT_LENGTH = 2000  
FRAUD_KEYPHRASES = [
    # Russian
    "вы выиграли", "переведите деньги", "номер карты", "код из смс",
    "cvv", "подтвердите платеж", "инвестиция", "гарантированная прибыль",
    "без риска", "вознаграждение", "комиссия", "скачайте приложение",
    "ваш счет заблокирован", "служба безопасности банка", "срок действия карты",
    "потеряете деньги", "лям", "миллион", "крупная сумма", "код подтверждения",
    "скажите код", "отправьте код", "скажите цифры",

    # English
    "you won", "transfer money", "send money", "verify your account", "click the link",
    "one-time code", "sms code", "card number", "cvv", "confirm payment",
    "guaranteed profit", "no risk", "commission", "download the app",
    "your account is blocked", "security service", "investment opportunity",
    "big money", "million", "lottery", "claim your prize", "urgent transfer",
    "wire transfer", "bank fee"
]
#patterns ti identifying sensitive info
REGEX_PATTERNS = {
    "card_number": r"\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{13,19}\b",
    "phone": r"\b(?:\+?\d{1,3}[-\s]?)?(?:[\(]?\d{3}[\)]?[-\s]?)?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b|\b(?:7|8)[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b",
    "possible_otp_code": r"\b\d{4,6}\b",
    "url_like": r"(https?://\S+|\bwww\.\S+\b)"
}

RISK_LEVELS = {
    "EXTREMELY_HIGH": {"en": "EXTREMELY HIGH", "ru": "КРАЙНЕ ВЫСОКИЙ"},
    "CRITICAL": {"en": "CRITICAL", "ru": "КРИТИЧЕСКИЙ"},
    "HIGH": {"en": "HIGH", "ru": "ВЫСОКИЙ"},
    "MEDIUM": {"en": "MEDIUM", "ru": "СРЕДНИЙ"},
    "LOW": {"en": "LOW", "ru": "НИЗКИЙ"},
}



def contains_cyrillic(s: str) -> bool:
    return bool(re.search('[\u0400-\u04FF]', s))

def choose_language_by_sample(recognizer: sr.Recognizer, audio_data: sr.AudioData) -> str:

    try:
        text_ru = recognizer.recognize_google(audio_data, language="ru-RU")
        if contains_cyrillic(text_ru):
            logging.info("Language heuristic: detected Russian from quick sample.")
            return "ru-RU"
    except Exception:
        pass  

    try:
     
        text_en = recognizer.recognize_google(audio_data, language="en-US")
        if not contains_cyrillic(text_en):
            logging.info("Language heuristic: detected English from quick sample.")
            return "en-US"
    except Exception:
        pass  
    logging.info("Language heuristic: defaulting to English (en-US).")
    return "en-US" 

def analyze_sentiment(text: str) -> Dict[str, Any]:
    if not text or tokenizer is None or model is None:
        return {"sentiment": "AI_ERROR", "score": 0.0}
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        sentiment = model.config.id2label.get(predicted_index, "UNKNOWN")
        score = probabilities[predicted_index].item()
        
        return {"sentiment": sentiment, "score": round(score, 4)}
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        return {"sentiment": "AI_ERROR", "score": 0.0}

def analyze_text_for_fraud(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
        
    low = text.lower()
    findings = []

    for phrase in FRAUD_KEYPHRASES:
        if phrase in low:
            findings.append({"type": "keyword (exact)", "phrase": phrase})
            continue 
        
        score_token = fuzz.token_set_ratio(phrase, low)
        if score_token >= FUZZY_THRESHOLD:
            findings.append({"type": "keyword (fuzzy)", "phrase": phrase, "score": round(score_token, 2)})
            continue
            
        score_partial = fuzz.partial_ratio(phrase, low)
        if score_partial >= FUZZY_THRESHOLD:
            findings.append({"type": "keyword (partial)", "phrase": phrase, "score": round(score_partial, 2)})

    for name, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            value_display = m.group(0)
            if len(value_display) > 50:
                value_display = value_display[:50] + "..."
            findings.append({"type": "regex", "pattern": name, "value": value_display})

    unique_findings = []
    seen = set()
    for f in findings:
        key = (f["type"], f.get("phrase", ""), f.get("pattern", ""), f.get("value", ""))
        if key not in seen:
            unique_findings.append(f)
            seen.add(key)

    return unique_findings

def get_risk_level_key(findings: List[Dict[str, Any]], sentiment_result: Dict[str, Any]) -> str:
    num_findings = len(findings)
    risk_score = 0

    if num_findings > 5:
        risk_score += 15
    elif num_findings > 2:
        risk_score += 10
    elif num_findings > 0:
        risk_score += 5

    sentiment = sentiment_result.get('sentiment')
    score = sentiment_result.get('score', 0)
    
    high_risk_sentiments = ['anger', 'fear', 'joy', 'surprise']
    if sentiment in high_risk_sentiments and score > 0.7:
        risk_score += 7
        
    if any(f.get("pattern") in ("card_number", "possible_otp_code") for f in findings):
         risk_score += 5

    if risk_score > 20:
        return "EXTREMELY_HIGH"
    elif risk_score > 12:
        return "CRITICAL"
    elif risk_score > 7:
        return "HIGH"
    elif risk_score > 0:
        return "MEDIUM"
    else:
        return "LOW"


def process_audio_data(audio_data: bytes, file_format: str) -> Dict[str, Any]:

    wav_path = None
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=file_format)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            audio_segment.export(tmpfile.name, format="wav")
            wav_path = tmpfile.name


        rec = sr.Recognizer()
        transcript = ""
        lang_guess = "en-US" 
        

        with sr.AudioFile(wav_path) as src_sample:
            try:
                sample = rec.record(src_sample, duration=6)
                lang_guess = choose_language_by_sample(rec, sample)
            except Exception:
                pass 
            
       
        with sr.AudioFile(wav_path) as src_full:
            audio = rec.record(src_full) 

       
        try:
            transcript = rec.recognize_google(audio, language=lang_guess)
        except sr.UnknownValueError:
            fallback_lang = "ru-RU" if lang_guess == "en-US" else "en-US"
            try:
                transcript = rec.recognize_google(audio, language=fallback_lang)
                lang_guess = fallback_lang
            except Exception:
                try:
                    transcript = rec.recognize_google(audio)
                except Exception as e:
                    logging.error(f"Final transcription failed: {e}")
                    return {"error": "Speech recognition failed. Try clearer audio."}

        findings = analyze_text_for_fraud(transcript)
        sentiment_result = analyze_sentiment(transcript)
        

        risk_key = get_risk_level_key(findings, sentiment_result)
        final_risk_en = RISK_LEVELS[risk_key]['en']
        final_risk_ru = RISK_LEVELS[risk_key]['ru']

        return {
            "transcript": transcript,
            "language": lang_guess,
            "findings": findings,
            "ai_analysis": sentiment_result,
            "risk_level_en": final_risk_en,
            "risk_level_ru": final_risk_ru
        }

    except FileNotFoundError:
        logging.error("FFMPEG not found. Please install ffmpeg and add it to PATH.")
        return {"error": "Audio conversion failed (ffmpeg/libav not found). Please install ffmpeg."}
    except sr.UnknownValueError:
        return {"error": "Could not recognize speech. Try a clearer audio sample."}
    except Exception as e:
        logging.error(f"Internal processing error: {e}", exc_info=True)
        return {"error": f"Internal error during processing: {e}"}
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)




async def send_long_text_as_pre(bot, chat_id: int, text: str):
    
    max_len = 4000  
    for i in range(0, len(text), max_len):
        chunk = text[i:i + max_len]
        await bot.send_message(chat_id=chat_id, text=f"<pre>{escape(chunk)}</pre>", parse_mode='HTML')

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hello! I'm a fraud-detection bot.\n"
        "Send a voice message or audio file (MP3, OGG, M4A) and I'll check it for scam indicators.\n\n"
        "Привет! Я бот-анализатор мошенничества.\n"
        "Отправь голосовое сообщение или аудиофайл (MP3, OGG, M4A), и я проверю его на признаки мошенничества.",
        parse_mode='HTML'
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    file_id = None
    file_format = None

    if message.voice:
        file_id = message.voice.file_id
        file_format = "ogg" 
    elif message.audio:
        file_id = message.audio.file_id
        file_format = message.audio.file_name.split('.')[-1].lower() if message.audio.file_name else "mp3"
    else:
        await message.reply_text("Send voice or audio file. / Отправьте голос или аудиофайл.")
        return

    await message.reply_text("Downloading and analyzing audio. This may take up to a minute.\nСкачиваю и анализирую аудио. Это может занять до минуты...")

    try:
        tg_file = await context.bot.get_file(file_id)
        with io.BytesIO() as buffer:
            await tg_file.download_to_memory(buffer)
            file_bytes = buffer.getvalue()
    except Exception as e:
        await message.reply_text(f"Download error: {e}\nОшибка скачивания: {e}")
        logging.error(f"Download error: {e}")
        return

    result = process_audio_data(file_bytes, file_format)

    if "error" in result:
        await message.reply_text(f"Error: {result['error']}\nОшибка: {result['error']}")
        return

    ai_status_en = "AI Sentiment: Unavailable (model failed)"
    ai_status_ru = "Сентимент AI: Недоступен (модель не загружена)"
    if result['ai_analysis']['sentiment'] != "AI_ERROR":
        ai_s = result['ai_analysis']['sentiment']
        ai_sc = result['ai_analysis']['score']
        ai_status_en = f"AI Sentiment: {ai_s} (Confidence: {ai_sc:.2f})"
        ai_status_ru = f"Сентимент AI: {ai_s} (уверенность: {ai_sc:.2f})"


    findings_lines = []
    for f in result['findings']:
        if f.get("type") == "regex":
            line = f"- {f['type'].upper()}: {escape(f.get('pattern'))} -> {escape(f.get('value'))}"
        else:
            line = f"- {f['type'].upper()}: {escape(f.get('phrase'))}"
        findings_lines.append(line)


    if findings_lines:
        findings_text_en = "\n".join(findings_lines)
        findings_text_ru = findings_text_en 
    else:
        findings_text_en = "— No fraud indicators found."
        findings_text_ru = "— Признаки мошенничества не обнаружены."


    transcript_full = result['transcript'] or ""
    transcript_snippet = escape(transcript_full[:300]) + ("..." if len(transcript_full) > 300 else "")


    header_en = (
        "<b>ANALYSIS RESULT</b>\n\n"
        f"<b>Risk Level:</b> {result['risk_level_en']}\n"
        f"<b>Detected Signs:</b> {len(result['findings'])}\n"
        f"{ai_status_en}\n\n"
        f"<b>Detected Indicators:</b>\n<pre>{findings_text_en}</pre>\n\n"
        f"<b>Transcript (start):</b>\n<pre>{transcript_snippet}</pre>\n"
    )

    header_ru = (
        "<b>РЕЗУЛЬТАТ АНАЛИЗА</b>\n\n"
        f"<b>Уровень риска:</b> {result['risk_level_ru']}\n"
        f"<b>Обнаружено признаков:</b> {len(result['findings'])}\n"
        f"{ai_status_ru}\n\n"
        f"<b>Найденные признаки:</b>\n<pre>{findings_text_ru}</pre>\n\n"
        f"<b>Транскрипт (начало):</b>\n<pre>{transcript_snippet}</pre>\n"
    )

    try:
        await message.reply_text(header_en, parse_mode='HTML')
        await message.reply_text(header_ru, parse_mode='HTML')

        if transcript_full:
            transcript_to_send: str
            label_en: str

            if len(transcript_full) > MAX_TRANSCRIPT_LENGTH:
                transcript_to_send = transcript_full[:MAX_TRANSCRIPT_LENGTH] + "\n\n... [Transcript truncated / Транскрипт урезан] ..."
                label_en = f"<b>Full Transcript (Truncated to {MAX_TRANSCRIPT_LENGTH} chars):</b>\n"
            else:
                transcript_to_send = transcript_full
                label_en = "<b>Full Transcript (EN/RU):</b>\n"

            await context.bot.send_message(chat_id=message.chat_id, text=label_en, parse_mode='HTML')
            await send_long_text_as_pre(context.bot, message.chat_id, transcript_to_send)

    except Exception as e:
        logging.error(f"Failed to send report: {e}")
        await message.reply_text(f"Error sending report: {e}")


def main() -> None:
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or not BOT_TOKEN:
        logging.error("BOT_TOKEN not set. Please set it in your .env file or environment.")
        sys.exit(1)

    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    logging.info("Bot is starting. Listening for messages...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
