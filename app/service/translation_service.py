from ..schema import TranslationRequest
# from ..model import mt5_model
from ..model import indic_trans_indic_en_model, indictrans_en_indic_model


class TranslationService:

    lang_dict = {
        'HINDI': 'hin_Deva',
        'ENGLISH': 'eng_Latn',
        'MARATHI': 'mr'
    }

    def translate(self, translation_request: TranslationRequest):
        text = translation_request.text
        src_lang = self.lang_dict.get(translation_request.source_language)
        target_lang = self.lang_dict.get(translation_request.target_language)

        # return mt5_model.translate(text, src_lang, target_lang)

        if src_lang == 'eng_Latn':
            return indictrans_en_indic_model.translate(text, src_lang, target_lang)
        else:
            return indic_trans_indic_en_model.translate(text, src_lang, target_lang)


translation_service = TranslationService()