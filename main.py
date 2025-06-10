import os
import torch
import asyncio
from PIL import Image
from torchvision import transforms, models
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ConversationHandler, ContextTypes
)
from utils.predict_crop import predict_crop
from utils.predict_disease import predict_disease
from chat_handler import answer_query as answer_model
from utils.chat_knowledge import build_knowledge_base


# Leaf Identification 
# Load Model
leaf_identifier = models.resnet18(pretrained=False)
leaf_identifier.fc = torch.nn.Linear(leaf_identifier.fc.in_features, 2)  
leaf_identifier.load_state_dict(torch.load(os.path.join("models", "leaf_identifier.pth"), map_location=torch.device('cpu')))
leaf_identifier.eval()

# Image transform 
leaf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load environment variables from .env file
load_dotenv('bot_token.env')
BOT_TOKEN = os.getenv("API_TOKEN")

# Constants
CROP_INPUT = 1
PHOTO_INPUT = 2
ASK_INPUT = 3


# /start command 
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌾 Welcome to FarmBot! \n\n"
        "Use:\n"
        "/recommend_crop - Get best crop suggestion\n"
        "/detect_disease - Detect plant disease\n"
        "/ask - Ask any farming-related question\n"
        "/cancel - Cancel any action"
    )

#  /cancel command
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Operation cancelled. Type /start to begin again.")
    return ConversationHandler.END

#  Crop Recommendation Flow 
async def recommend_crop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Please enter the following 7 values separated by commas:\n"
        "N, P, K, Temperature, Humidity, pH, Rainfall\n\nExample:\n80, 40, 30, 25, 70, 6.5, 200",
        parse_mode="Markdown"
    )
    return CROP_INPUT

async def crop_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Parse input values 
        values = list(map(float, update.message.text.split(',')))

        if len(values) != 7:
            raise ValueError("Exactly 7 values required.")
        
        # Extract individual values
        N, P, K, Temperature, Humidity, pH, Rainfall= values

        # Perform the realistic value range checks
        if not (0 <= Rainfall <= 3000):
            raise ValueError("Rainfall must be between 200 mm and 3000 mm.")
        if not (0 <= Temperature <= 50):
            raise ValueError("Temperature must be between 0 °C and 50 °C.")
        if not (10 <= N <= 300):
            raise ValueError("N (Nitrogen) should be between 10 and 300 kg/ha.")
        if not (5 <= P <= 200):
            raise ValueError("P (Phosphorus) should be between 5 and 200 kg/ha.")
        if not (10 <= K <= 200):
            raise ValueError("K (Potassium) should be between 10 and 200 kg/ha.")
        if not (0 <= Humidity <= 100):
            raise ValueError("Humidity should be between 0% and 100%.")
        if not (5 <= pH <= 9.5):
            raise ValueError("pH should be between 5 and 9.5 .")

        # If all values are within bounds, proceed to prediction
        crop = predict_crop(*values)
        await update.message.reply_text(f"Recommended Crop: *{crop}*", parse_mode="Markdown")

        # After prediction, prompt the user again without ending the conversation
        await update.message.reply_text(
            "You can enter another 7 values for new crop recommendation, or type /cancel to stop.",
            parse_mode="Markdown"
        )
        return CROP_INPUT  # Stay in the same state

    except Exception as e:
        await update.message.reply_text(f"⚠️ Invalid input: {str(e)}. Please try again.")
    return CROP_INPUT  

#  Disease Detection Flow
async def detect_disease_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please send a clear image of the leaf.")
    return PHOTO_INPUT

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        await update.message.reply_text("⚠️ Please send a valid photo.")
        return

    photo = update.message.photo[-1]
    file = await photo.get_file()
    img_path = f"temp_{update.message.from_user.id}.jpg"
    await file.download_to_drive(img_path)

    try:
        # Leaf or Non-leaf check first
        img = Image.open(img_path).convert("RGB")
        img_tensor = leaf_transform(img).unsqueeze(0)

        with torch.no_grad():
            output = leaf_identifier(img_tensor)
            pred = torch.argmax(output, 1).item()

        if pred == 1:  # Non-leaf
            await update.message.reply_text("This doesn't seem to be a leaf. Please send a proper leaf image.")
        else:
            # Proceed with Disease Detection
            result = predict_disease(img_path)
            await update.message.reply_text(f"Detected Disease: *{result}*", parse_mode="Markdown")

            # After prediction, ask again without ending
            await update.message.reply_text(
                "You can upload another image for new disease detection, or type /cancel to stop.",
                parse_mode="Markdown"
            )
        return PHOTO_INPUT  # Stay in the same state


    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}")
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

# Chat Flow 
async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please type your farming/crop related question:")
    return ASK_INPUT  # return the state

async def handle_user_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    await update.message.reply_chat_action("typing")
    await asyncio.sleep(1)

    try:
        answer = answer_model(user_question)
        await update.message.reply_text(f"*Answer:*\n{answer}", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error fetching answer: {str(e)}")
    return ASK_INPUT  

#  Main Function 
def main():
    if not (os.path.exists("knowledge_base/farming_knowledge.faiss") and os.path.exists("knowledge_base/farming_knowledge.pkl")):
        build_knowledge_base()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    # Ask Conversation
    ask_conv = ConversationHandler(
        entry_points=[CommandHandler("ask", ask_command)],
        states={ASK_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_question)]},
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    app.add_handler(ask_conv)

    # Crop Conversation
    crop_conv = ConversationHandler(
        entry_points=[CommandHandler("recommend_crop", recommend_crop)],
        states={CROP_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex(r'^(\d+\.?\d*,\s*){6}\d+\.?\d*$'),
                crop_input)]},
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    app.add_handler(crop_conv)

    # Disease Detection Conversation
    disease_conv = ConversationHandler(
        entry_points=[CommandHandler("detect_disease", detect_disease_command)],
        states={PHOTO_INPUT: [MessageHandler(filters.PHOTO, image_handler)]},
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    app.add_handler(disease_conv)

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Bot crashed: {e}")
        raise



