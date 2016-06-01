import telebot
import config
import urllib.request
import subprocess
import os

bot = telebot.TeleBot(config.TOKEN);

@bot.message_handler(commands = ['start', 'help'])
def start_message(message):
    bot.reply_to(message, "I will search a movie for you. Just send me an image")

@bot.message_handler(func = lambda m: m.text == "Who will win?")
def champions_league(message):
    bot.reply_to(message, "Chelsea! Chelsea! Chelsea! Keep the blue flag flying high!:)")


def get_answer(image_path):
    result = subprocess.check_output(["/home/oracle/.CLion2016.1/system/cmake/generated/comvision-dd4ca9a3/dd4ca9a3/Release/comvision /home/oracle/Project/bot/" + image_path], shell = True)
    return result

@bot.message_handler(content_types = ['document'])
def file_error(message):
    bot.reply_to(message, 'Use "Photo" instead of "File".')	
	
@bot.message_handler(content_types = ['photo'])
def recognize_photo(message):
    image_path = str(message.from_user.id) + ".jpg"

    if (os.path.isfile(image_path)):
        bot.reply_to(message, "Try again later.")
        return

    bot.reply_to(message, "This can take a bit of time, wait patiently, please.")
    biggest_photo = message.photo[0]
    for photo in message.photo:
        if (photo.width > biggest_photo.width):
            biggest_photo = photo

    photo_file = bot.get_file(biggest_photo.file_id)
    full_path = config.BASE_PATH + config.TOKEN + "/" + photo_file.file_path
    print(full_path)

    with urllib.request.urlopen(full_path) as f:
        image = open(image_path, mode = "wb")
        image.write(f.read())
        image.close()
    bot.reply_to(message, get_answer(str(message.from_user.id) + ".jpg"))

    os.remove(image_path)

subprocess.call(["rm -rf *.jpg"], shell = True)
bot.polling()

