import json
import torch
# # Text changes that shouldn't change meaning but might
# from nlaugmenter.transformations.english_inflectional_variation.transformation import EnglishInflectionalVariation
# from nlaugmenter.transformations.filler_word_augmentation.transformation import FillerWordAugmentation
# from nlaugmenter.transformations.hashtagify.transformation import Hashtagify
# from nlaugmenter.transformations.speech_disfluency_perturbation.transformation import SpeechDisfluencyPerturbation
# from nlaugmenter.transformations.tense.transformation import Tense


# Paraphrasing. Could change syntax + meaning
from nlaugmenter.transformations.back_translation.transformation import BackTranslation
from nlaugmenter.transformations.diverse_paraphrase.transformation import DiverseParaphrase
from nlaugmenter.transformations.protaugment_diverse_paraphrase.transformation import ProtaugmentDiverseParaphrase

# TODO: Add two more paraphrase methods if time permits


# Rule-based syntactic changes
from nlaugmenter.transformations.propbank_srl_roles.transformation import CheckSrl as PropbankSrlRoles

# Semantic word-based changes that likely won't change syntax
from nlaugmenter.transformations.replace_with_hyponyms_hypernyms.transformation import ReplaceHypernyms
from nlaugmenter.transformations.replace_with_hyponyms_hypernyms.transformation import ReplaceHyponyms
from nlaugmenter.transformations.slangificator.transformation import Slangificator
from nlaugmenter.transformations.synonym_substitution.transformation import SynonymSubstitution

# Setup paraphrasing transformations
backtranslation = BackTranslation()
diverse_paraphrase = DiverseParaphrase()
protaugment_diverse_paraphrase = ProtaugmentDiverseParaphrase()


PARAPHRASING_TRANSFORMATIONS = {
    "backtranslation": lambda sentence: backtranslation.generate(sentence),
    "diverseparaphrase": lambda sentence: diverse_paraphrase.generate(sentence),
    "protaugmentdiverseparaphrase": lambda sentence: protaugment_diverse_paraphrase.generate(sentence)
    }

# Setup rule-based transformations
propbank_srl_roles = PropbankSrlRoles(max_outputs=3)

RULEBASED_TRANSFORMATIONS = {
    "propbanksrlroles": lambda sentence: propbank_srl_roles.generate(sentence)
    }

# Setup semantic word-based transformations
replace_hyponyms = ReplaceHyponyms()
replace_hypernyms = ReplaceHypernyms()
slangificator = Slangificator(max_outputs=3)
synonym_substitution = SynonymSubstitution(max_outputs=3)


SEMANTIC_WORDBASED_TRANSFORMATIONS = {
    "replacehyponyms": lambda sentence: replace_hyponyms.generate(sentence),
    "replacehypernyms": lambda sentence: replace_hypernyms.generate(sentence),
    "slangificator": lambda sentence: slangificator.generate(sentence),
    "synonymsubstitution": lambda sentence: synonym_substitution.generate(sentence)
    }

TRANSFORMATIONS = {
    "paraphrasing": PARAPHRASING_TRANSFORMATIONS,
    "rulebased": RULEBASED_TRANSFORMATIONS,
    "semanticwordbased": SEMANTIC_WORDBASED_TRANSFORMATIONS,
    }

def generate_variants(sentence):
    """
    Generate variants of a sentence.
    """
    variants = []
    variants.append({"type": "original", "name": "original", "variant": sentence})

    for transform_type, v in TRANSFORMATIONS.items():
        for transform_name, transformation in v.items():
            try:
                generation = transformation(sentence)
                for variant in generation:
                    if variant == '' or variant == sentence:
                        continue
                    variants.append({"type": transform_type,
                                    "name": transform_name,
                                    "variant": variant})
            except Exception as e:
                continue
    
    return variants
    
if __name__ == "__main__":
    with open("dataset/examples.jsonl", 'r') as f:
        text_data = [json.loads(line.strip()) for line in f]
    
    with open("dataset/examples_augmented.jsonl", 'w', buffering=1) as f:
        for example in text_data:
            example["caption_0"] = generate_variants(example["caption_0"])
            example["caption_1"] = generate_variants(example["caption_1"])
            print(json.dumps(example), file=f)