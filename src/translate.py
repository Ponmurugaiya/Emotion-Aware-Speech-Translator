from googletrans import Translator

def translate_text(text):
    """ Translates Tamil text to English """
    translator = Translator()
    translated = translator.translate(text, dest="en")
    return translated.text
