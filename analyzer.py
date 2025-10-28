import sys
import os
import subprocess
import re
import logging
import io
import tempfile
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
load_dotenv()



# --- Настройка логирования и УСТАНОВКА ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Словарь для сопоставления имени установки (pip) и имени импорта (import)
PACKAGE_MAPPING: List[Tuple[str, str]] = [
    ("speech_recognition", "speech_recognition"),
    ("pydub", "pydub"),
    ("rapidfuzz", "rapidfuzz"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("python-telegram-bot", "telegram"), 
]

def install_and_import(package_list: List[Tuple[str, str]]):
    logging.info("Проверяю и устанавливаю необходимые библиотеки...")
    for pip_name, import_name in package_list:
        try:
            __import__(import_name)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "-q", pip_name])
                logging.info(f"Установлен: {pip_name}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Не удалось установить {pip_name}: {e}")
                sys.exit(1)
        try:
             __import__(import_name)
        except Exception as e:
             logging.error(f"Не удалось импортировать {import_name} после установки. Критическая ошибка: {e}")
             sys.exit(1)

# Вызов функции с новым списком
install_and_import(PACKAGE_MAPPING)

import speech_recognition as sr
from pydub import AudioSegment
from rapidfuzz import fuzz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from html import escape # Импортируем для экранирования HTML-символов

# --- НАСТРОЙКА AI-МОДЕЛИ И ТОКЕНА ---
# !!! ЗАМЕНИТЕ ЭТОТ ТОКЕН НА ВАШ АКТУАЛЬНЫЙ !!!
BOT_TOKEN = os.getenv("BOT_TOKEN") or "ВАШ_ТОКЕН_БОТА_ЗДЕСЬ"

MODEL_NAME = "cointegrated/rubert-tiny2-cedr-emotion-detection" 
tokenizer = None
model = None


def load_ai_model():
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        logging.info("AI-модель для анализа тональности успешно загружена.")
    except Exception as e:
        logging.error(f"НЕ УДАЛОСЬ ЗАГРУЗИТЬ AI-МОДЕЛЬ (NLP). AI-АНАЛИЗ БУДЕТ НЕДОСТУПЕН. Причина: {e}")
        tokenizer = None
        model = None

# Загружаем модель перед запуском бота
load_ai_model()


# --- ФРАЗЫ И ПАТТЕРНЫ ---
FUZZY_THRESHOLD = 70
FRAUD_KEYPHRASES = [
    "вы выиграли", "переведите деньги", "номер карты", "код из смс",
    "cvv", "подтвердите платеж", "инвестиция", "гарантированная прибыль",
    "без риска", "вознаграждение", "комиссия", "скачайте приложение",
    "ваш счет заблокирован", "служба безопасности банка", "срок действия карты",
    "click the link", "transfer money", "verify your account", "потеряете деньги",
    "лям", "миллион", "крупная сумма", "код подтверждения",
    "скажите код", "отправьте код", "скажите цифры"
]
REGEX_PATTERNS = {
    "card_number": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b|\b(?:\d[ -]*?){13,19}\b",
    "phone": r"\b(?:7|8)[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b|\b(\+?\d{1,3}[-\s]?)?(\(?\d{3,4}\)?[-\s]?)?\d{3,4}[-\s]?\d{2,4}\b",
    "otp_like": r"\b\d{4,6}\b"
}

# --- ФУНКЦИИ АНАЛИЗА (Без изменений) ---

def analyze_sentiment(text: str) -> Dict[str, Any]:
    if not text or tokenizer is None or model is None: 
        return {"sentiment": "AI_ERROR", "score": 0.0}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    labels = model.config.id2label
    predicted_index = torch.argmax(probabilities).item()
    sentiment = labels.get(predicted_index, "UNKNOWN")
    score = probabilities[predicted_index].item()
    return {"sentiment": sentiment, "score": round(score, 4)}

def analyze(text: str) -> List[Dict[str, Any]]:
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
        for m in re.finditer(pat, text):
            value_display = m.group(0)[:50] + "..." if len(m.group(0)) > 50 else m.group(0)
            findings.append({"type": "regex", "pattern": name, "value": value_display})
            
    unique_findings = []
    seen = set()
    for f in findings:
        key = (f["type"], f.get("phrase", ""), f.get("pattern", ""), f.get("value", ""))
        if key not in seen:
            unique_findings.append(f)
            seen.add(key)
            
    return unique_findings

def determine_risk(findings: List[Dict[str, Any]], sentiment_result: Dict[str, Any]) -> str:
    num_findings = len(findings)
    risk_score = 0 
    
    if num_findings > 5:
        risk_score += 15
    elif num_findings > 2:
        risk_score += 10
    elif num_findings > 0:
        risk_score += 5
        
    if sentiment_result.get('sentiment') == 'NEGATIVE' and sentiment_result.get('score', 0) > 0.8:
        risk_score += 7
        
    if risk_score > 20:
        return "КРАЙНЕ ВЫСОКИЙ (AI-Оценка)"
    elif risk_score > 12:
        return "КРИТИЧЕСКИЙ"
    elif risk_score > 7:
        return "ВЫСОКИЙ"
    elif risk_score > 0:
        return "СРЕДНИЙ"
    else:
        return "НИЗКИЙ"

# --- ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ АУДИО ---

def process_audio_data(audio_data: bytes, file_format: str) -> Dict[str, Any]:
    wav_path = None
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=file_format)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            audio_segment.export(tmpfile.name, format="wav")
            wav_path = tmpfile.name
        
        rec = sr.Recognizer()
        with sr.AudioFile(wav_path) as src:
            audio = rec.record(src)
        
        text = rec.recognize_google(audio, language="ru-RU")
        
        findings = analyze(text)
        sentiment_result = analyze_sentiment(text)
        final_risk = determine_risk(findings, sentiment_result)
        
        return {
            "transcript": text,
            "findings": findings,
            "ai_analysis": sentiment_result,
            "risk_level": final_risk
        }
    except FileNotFoundError:
        return {"error": "Не найден ffmpeg/libav. Установите их и добавьте в PATH."}
    except sr.UnknownValueError:
        return {"error": "Не удалось распознать речь. Попробуйте более четкое аудио."}
    except Exception as e:
        logging.error(f"Ошибка обработки аудио: {e}")
        return {"error": f"Произошла внутренняя ошибка при обработке: {e}"}
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

# --- TELEGRAM HANDLERS  ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот-анализатор мошенничества.\n"
        "Просто отправь мне <b>голосовое сообщение</b> или <b>аудиофайл</b> (MP3, OGG, M4A) и я проверю его на признаки мошенничества с помощью AI.",
        parse_mode='HTML' # 
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
        return

    await message.reply_text("Получаю и начинаю анализ аудио. Это может занять до минуты...")
    
    try:
        tg_file = await context.bot.get_file(file_id)
        
        with io.BytesIO() as buffer:
            await tg_file.download_to_memory(buffer) 
            file_bytes = buffer.getvalue()

    except Exception as e:
        await message.reply_text(f"Ошибка при скачивании файла: {e}")
        logging.error(f"Ошибка скачивания: {e}")
        return

    result = process_audio_data(file_bytes, file_format)
    
    if "error" in result:
        await message.reply_text(f"Ошибка: {result['error']}")
        return

    # --- Формирование отчета с HTML ---
    
    ai_status = f"<b>AI Сентимент:</b> {result['ai_analysis']['sentiment']} (Уверенность: {result['ai_analysis']['score']:.2f})"
    if result['ai_analysis']['sentiment'] == 'AI_ERROR':
         ai_status = f"<b>AI Сентимент:</b> Недоступен (Проблема загрузки модели)"


    findings_text = "\n".join([
        f"- {f['type'].upper()}: <code>{escape(f.get('phrase', f.get('pattern')))}</code>"
        for f in result['findings']
    ]) or "— Признаки мошенничества не обнаружены."

    # Используем HTML теги <b>, <i> и <pre> (для блока кода)
    report = (
        f"<b>📊 РЕЗУЛЬТАТ АНАЛИЗА</b>\n\n"
        f"<b>УРОВЕНЬ РИСКА:</b> {result['risk_level']}\n"
        f"<b>Обнаружено признаков:</b> {len(result['findings'])}\n"
        f"{ai_status}\n\n"
        f"<b>🚨 Найденные Признаки:</b>\n"
        f"{findings_text}\n\n"
        f"<b>📝 Транскрипт (начало):</b>\n"
        f"<i>{escape(result['transcript'][:150])}...</i>\n\n" # Экранируем, чтобы не сломать теги
        f"<b>Полный транскрипт:</b>\n"
        f"<pre>{escape(result['transcript'])}</pre>" 
    )

    await message.reply_text(report, parse_mode='HTML') # 

# --- MAIN BOT RUNNER  ---
def main() -> None:
    if BOT_TOKEN == "ВАШ_ТОКЕН_БОТА_ЗДЕСЬ":
        logging.error("Пожалуйста, замените BOT_TOKEN на актуальный токен вашего Telegram-бота.")
        sys.exit(1)
        
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    logging.info("Бот запущен. Ожидание сообщений...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()