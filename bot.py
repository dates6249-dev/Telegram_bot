from transformers import AutoTokenizer, AutoModelForCausalLM
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# ================== НАСТРОЙКИ ==================

TELEGRAM_TOKEN = "8343418132:AAFzbuY_fBnh4LmP_DqTwvEtWeCfEUUe3l4"
HUGGINGFACE_TOKEN = "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"  # Не забудь заменить на свой API-ключ Hugging Face

# ================== ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА ==================

model_name = "sberbank-ai/rugpt3small_based_on_gpt2"  # Пример для ruGPT-3
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ================== СТАРТ ==================

def start(update: Update, context: CallbackContext):
    """Приветственное сообщение при старте бота"""
    update.message.reply_text(
        "Привет! Я здесь, чтобы поддержать тебя. Можешь рассказать, что тебя беспокоит? Я постараюсь выслушать и помочь."
    )

# ================== ОБРАБОТКА СООБЩЕНИЙ ==================

def handle_message(update: Update, context: CallbackContext):
    """Обработка сообщений с естественной поддержкой и беседой"""
    user_text = update.message.text.strip()

    if not user_text:
        update.message.reply_text("Ты хочешь что-то обсудить? Я слушаю!")
        return

    try:
        # Создаём запрос для модели с естественным тоном
        prompt = f"Ты хороший собеседник, который поддерживает человека. Ответь спокойно, поддерживай собеседника, задавай вопросы, чтобы разговор был живым и естественным. Не давай сухих советов, лучше поговори с ним, как настоящий друг.\n\nВопрос: {user_text}\nОтвет:"

        inputs = tokenizer.encode(prompt, return_tensors="pt")

        # Генерация ответа с естественным подходом
        outputs = model.generate(
            inputs,
            max_length=150,  # Увеличиваем длину ответа, чтобы сделать разговор более полным
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.8,  # Средняя температура для вариативных, но адекватных ответов
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

        # Декодируем ответ
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Если ответ слишком короткий или не совсем понятный, уточняем
        if len(answer.split()) < 5:
            answer = "Прости, я не совсем понял. Можешь рассказать немного больше?"

        # Отправляем ответ
        update.message.reply_text(answer)

    except Exception as e:
        # Обработка ошибок
        update.message.reply_text(f"Ой, что-то пошло не так. Пожалуйста, попробуй позже. Ошибка: {e}")

# ================== ЗАПУСК БОТА ==================

def main():
    """Основная функция для запуска бота"""
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Обработчики команд и сообщений
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # Запуск бота
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

