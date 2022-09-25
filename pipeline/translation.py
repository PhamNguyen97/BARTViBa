from GraphTranslation.services.base_service import BaseServiceSingleton
from GraphTranslation.pipeline.translation import TranslationPipeline, Languages, TranslationGraph
from pipeline.model_translate import ModelTranslator
from GraphTranslation.common.ner_labels import *
import time


class Translator(BaseServiceSingleton):
    def __init__(self):
        super(Translator, self).__init__()
        self.model_translator = ModelTranslator()
        self.graph_translator = TranslationPipeline()
        self.graph_translator.eval()

    def __call__(self, text: str, model: str = "BART_CHUNK"):
        if model in ["BART_CHUNK", "BART_CHUNK_NER_ONLY"]:
            # nlp_core_time = time.time()
            sentence = self.graph_translator.nlp_core_service.annotate(text, language=Languages.SRC)
            sentence = self.graph_translator.graph_service.add_info_node(sentence)
            translation_graph = TranslationGraph(src_sent=sentence)
            translation_graph.update_src_sentence()
            if model == "BART_CHUNK":
                mapped_words = [w for w in translation_graph.src_sent if len(w.translations) > 0 or w.is_ner]
            else:
                mapped_words = [w for w in translation_graph.src_sent if w.is_punctuation or w.is_ner]
            # print(f"NLP CORE TIME {time.time() - nlp_core_time}")
            result = []
            src_mapping = []
            i = 0
            while i < len(mapped_words) - 1:
                src_from_node = mapped_words[i]
                if src_from_node.is_ner:
                    # ner_time = time.time()
                    ner_text = self.graph_translator.translate_ner(src_from_node.original_upper)
                    if src_from_node.ner_label in [NUM]:
                        result.append(ner_text.lower())
                    else:
                        result.append(ner_text)
                    # print(
                    #     f"NER {src_from_node.original_upper} >> {ner_text}: {src_from_node.ner_label} : {time.time() - ner_time}")
                else:
                    translations = src_from_node.translations
                    if len(translations) == 1:
                        result.append(translations[0].text)
                        # print(f"WORD {src_from_node.text} : {translations[0].text}")
                    else:
                        result.append(translations)
                        # print(f"WORD {src_from_node.text} : {[item.text for item in src_from_node.translations]}")

                    # word = src_from_node.translations[0]
                    # word.has_next_word()
                    # word.has_last_word()
                src_mapping.append([src_from_node])
                src_to_node = mapped_words[i + 1]
                if src_from_node.end_index < src_to_node.begin_index - 1:
                    chunk = translation_graph.src_sent.get_chunk(src_from_node.begin_index + 1,
                                                                 src_to_node.end_index - 1)
                    # chunk_time = time.time()
                    # translated_chunk = chunk.text
                    translated_chunk = self.model_translator.translate_cache(chunk.text)
                    result.append(translated_chunk)
                    # print(f"MODEL TRANSLATE {chunk} >>> {translated_chunk}", time.time() - chunk_time)
                i += 1

            if len(result) >= 3:
                for i in range(len(result)):
                    if not isinstance(result[i], str):
                        scores = [0] * len(result[i])
                        if i > 0:
                            before_word = result[i - 1]
                        else:
                            before_word = None
                        if i < len(result) - 1 and isinstance(result[i + 1], str):
                            next_word = result[i + 1]
                        else:
                            next_word = None
                        if next_word is None and before_word is None:
                            result[i] = result[i][0]
                            continue
                        candidates = result[i]
                        max_score = 0
                        best_candidate = None
                        if before_word is not None:
                            before_word = before_word.split()[-1]
                            before_word = self.graph_translator.graph_service.graph \
                                .get_node_by_text(before_word, language=Languages.DST)
                        if next_word is not None:
                            next_word = next_word.split()[0]
                            next_word = self.graph_translator.graph_service.graph\
                                .get_node_by_text(next_word, language=Languages.DST)
                        for j, candidate in enumerate(candidates):
                            if before_word is not None and candidate.has_last_word(before_word, distance_range=(0, 3)):
                                scores[j] += 1
                            if next_word is not None and candidate.has_next_word(next_word, distance_range=(0, 3)):
                                scores[j] += 1
                            if scores[j] > max_score or best_candidate is None:
                                best_candidate = candidate
                                max_score = scores[j]
                        result[i] = best_candidate.text
                        print(f"word {best_candidate.text}: {max_score}")
            output = result
            output = "  ".join(output).replace("@", "")
            while "  " in output:
                output = output.replace("  ", " ")
            return output.strip()
        else:
            # model_time = time.time()
            output = self.model_translator.translate(text)
            # print(f"MODEL TIME {time.time() - model_time}")
            return output


if __name__ == "__main__":
    translator = Translator()
    print(translator("xin chào, tôi là sinh viên đại học Bách Khoa"))
