from google.cloud import translate_v2 as translate


class GoogleTranslator:
    def __init__(self):
        self.client = translate.Client()

    def translate(self, text, src='auto', dest='en'):
        if not text.strip():
            return ""

        result = self.client.translate(text, src=src, dest=dest)
        return result.text

google_translator = GoogleTranslator()
