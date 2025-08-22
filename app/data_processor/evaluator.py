from sacrebleu import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from pyter import ter


class Evaluator:
    # Download necessary NLTK data
    nltk.download('wordnet')
    nltk.download('omw-1.4')


    def evaluate_translations(hypotheses: list, references: list, verbose: bool = False) -> dict:
        assert len(hypotheses) == len(references), "Mismatched number of predictions and references."

        # BLEU
        bleu = corpus_bleu(hypotheses, [references])
        bleu_score = bleu.score

        # METEOR
        meteor_scores = [meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)]
        meteor_avg = sum(meteor_scores) / len(meteor_scores)

        # TER
        ter_scores = [ter(hyp, ref) for ref, hyp in zip(references, hypotheses)]
        ter_avg = sum(ter_scores) / len(ter_scores)

        # Results
        results = {
            "BLEU": round(bleu_score, 2),
            "METEOR": round(meteor_avg, 4),
            "TER": round(ter_avg, 4)
        }

        if verbose:
            print("=== Evaluation Metrics ===")
            print(f"BLEU Score : {results['BLEU']}")
            print(f"METEOR Score : {results['METEOR']}")
            print(f"TER Score : {results['TER']}")

        return results

evaluator = Evaluator()
