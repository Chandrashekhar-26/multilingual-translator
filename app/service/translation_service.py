from ..schema import TranslationRequest
from ..model import indic_trans_indic_en_model, indictrans_en_indic_model, indictrans_indic_indic_model


class TranslationService:

    lang_dict = {
        'HINDI': 'hin_Deva',
        'ENGLISH': 'eng_Latn',
        'MARATHI': 'mar_Deva',
        'TAMIL': 'tam_Taml',
        'KANNADA': 'kan_Knda'
    }

    def translate(self, translation_request: TranslationRequest):
        text = translation_request.text
        src_lang = self.lang_dict.get(translation_request.source_language)
        target_lang = self.lang_dict.get(translation_request.target_language)

        if src_lang == 'eng_Latn':
            return indictrans_en_indic_model.translate(text, src_lang, target_lang)
        elif target_lang == 'eng_Latn':
            return indic_trans_indic_en_model.translate(text, src_lang, target_lang)
        else:
            return indictrans_indic_indic_model.translate(text, src_lang, target_lang)


translation_service = TranslationService()