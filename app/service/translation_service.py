from ..schema import TranslationRequest
from ..model import indic_trans_indic_en_model, indictrans_en_indic_model, indictrans_indic_indic_model
from .google_translator import google_translator
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score


class TranslationService:

    lang_dict = {
        'HINDI': 'hin_Deva',
        'ENGLISH': 'eng_Latn',
        'MARATHI': 'mar_Deva',
        'TAMIL': 'tam_Taml',
        'KANNADA': 'kan_Knda'
    }

    def init_finetune(self):
        indic_trans_indic_en_model.train()
        indictrans_en_indic_model.train()
        indictrans_indic_indic_model.train()

    async def translate(self, translation_request: TranslationRequest):
        text = translation_request.text
        src_lang = self.lang_dict.get(translation_request.source_language)
        target_lang = self.lang_dict.get(translation_request.target_language)

        model_translation_result = ''

        if src_lang == 'eng_Latn':
            model_translation_result = indictrans_en_indic_model.translate(text, src_lang, target_lang)
        elif target_lang == 'eng_Latn':
            model_translation_result = indic_trans_indic_en_model.translate(text, src_lang, target_lang)
        else:
            model_translation_result = indictrans_indic_indic_model.translate(text, src_lang, target_lang)

        # google translation
        google_translation = await google_translator.translate(text, translation_request.source_language, translation_request.target_language)

        # evaluation
        metrics = self.get_metrics(model_translation_result, google_translation, translation_request.target_language)

        result = {
            'model_translation': model_translation_result,
            'google_translation': google_translation,
            'metrics': metrics
        }

        return result

    def get_metrics(self, model_translation, google_translation, target_lang):
        ref = [[google_translation]]
        hyp = [model_translation]

        metrics = {
            "bleu": None,
            "ter": None,
            "meteor": None
        }

        try:

            tokenize_type = 'intl' if target_lang not in ["ENGLISH", "en"] else '13a'
            bleu = sacrebleu.corpus_bleu(hyp, ref, tokenize=tokenize_type).score
            ter = sacrebleu.corpus_ter(hyp, ref).score

            meteor = None
            if target_lang == "en":
                tokenized_ref = [nltk.word_tokenize(google_translation)]
                tokenized_hyp = nltk.word_tokenize(model_translation)
                meteor = meteor_score(tokenized_ref, tokenized_hyp)

            metrics = {
                "bleu": round(bleu, 2),
                "ter": round(ter, 2),
                "meteor": round(meteor, 2) if meteor is not None else None
            }
        except Exception as e:
            metrics = {
                "bleu": None,
                "ter": None,
                "meteor": None
            }

        return metrics


translation_service = TranslationService()