from googletrans import Translator
import asyncio


class GoogleTranslator:
    client:Translator = None

    google_trans_map = {
        "auto": "auto",
        "AUTO": "auto",
        'HINDI': 'hi',
        'ENGLISH': 'en',
        'MARATHI': 'mr',
        'TAMIL': 'ta',
        'KANNADA': 'kn'
    }

    def __init__(self):
        self.client: Translator = Translator()

    async def _translate_async(self, text, src, dest):
        return await self.client.translate(text, src=src, dest=dest)

    async def translate(self, text, source_lang='auto', target_lang='en'):
        if not text.strip():
            return ""

        src_lng = self.google_trans_map.get(source_lang)
        tgt_lng = self.google_trans_map.get(target_lang)

        result = await self._translate_async(text, src_lng, tgt_lng)
        return result.text


google_translator = GoogleTranslator()
