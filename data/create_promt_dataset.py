import json
import os
from typing import Tuple, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from GraphTranslation.common.data_types import RelationTypes
from GraphTranslation.common.languages import Languages
from GraphTranslation.services.graph_service import GraphService
from custom_dataset.vi_ba_aligned_dataset import ViBaDataset
from objects.graph import (
    TranslationGraph,
    Sentence
)

graph_service = GraphService()


def preprocess(vi_sent, ba_sent) -> Tuple[TranslationGraph, List]:
    src_sentence = graph_service.nlp_core_service.annotate(
        text=vi_sent,
        language=Languages.SRC
    )
    dst_sentence = graph_service.nlp_core_service.annotate(
        text=ba_sent,
        language=Languages.DST
    )
    src_sentence = graph_service.add_info_node(src_sentence)
    dst_sentence = graph_service.add_info_node(dst_sentence)
    translation_graph = TranslationGraph(
        src_sent=src_sentence,
        dst_sent=dst_sentence
    )
    translation_graph, extra_relations = graph_service.find_anchor_parallel(translation_graph)
    return translation_graph, extra_relations


def mapping_real_entities(syllable_sent, word_based_sent):
    syllables = syllable_sent.words
    words = word_based_sent.words

    i = 0

    for word in words:
        word_length = len(word.text.split())
        j = 0
        while j < word_length:
            if i + j >= len(syllables):
                break

            if word.ner_label != "O":
                syllables[i + j].ner_label = (
                    word.ner_label.replace("B-", "").replace("I-", "")
                    if word.ner_label is not None
                    else "O"
                )
            syllable_word_length = len(syllables[i + j].text.split())
            word_length -= (syllable_word_length - 1)
            j += 1
        i += word_length


def ba_only_entity_prompt(vi_sent: Sentence, ba_sent: Sentence):
    entities = []
    for i, word in enumerate(ba_sent.words):
        if word.ner_label in ["O", None]:
            continue
        if not entities or entities[-1][0] < i - 1 or entities[-1][-1] != word.ner_label:
            entities.append([i, [word], word.ner_label])
        else:
            entities[-1][1].append(word)
            entities[-1][0] = i


def data_pipeline(vi_sent: str, ba_sent: str):
    translation_graph, extra_relations = preprocess(vi_sent=vi_sent, ba_sent=ba_sent)

    # mapping_entities
    src_sentence_with_entities = graph_service.nlp_core_service.annotate(
        text=vi_sent,
        language=Languages.SRC,
        is_train=False
    )

    mapping_real_entities(
        syllable_sent=translation_graph.src_sent,
        word_based_sent=src_sentence_with_entities
    )

    for word in translation_graph.src_sent.words:
        mapping_relations = word.get_relation_by_type(RelationTypes.MAPPING)
        if mapping_relations:
            mapping_relations[0].dst.ner_label = word.ner_label
        # print(word.text, word.ner_label, word.get_relation_by_type(RelationTypes.MAPPING))
    # for word in translation_graph.dst_sent.words:
    #     print(word.text, word.ner_label)
    # print()
    # print(translation_graph.dict)

    ba_only_entity_prompt(vi_sent=translation_graph.src_sent, ba_sent=translation_graph.dst_sent)
    ba_output = []
    for word in translation_graph.dst_sent.words:
        word_info = word.info
        word_info["begin"] -= 1
        word_info["end"] -= 1
        word_info.pop("pre")
        word_info.pop("pos")
        word_info.pop("next")
        word_info["ner"] = word_info["ner"] if word_info["ner"] else "O"
        ba_output.append(word_info)

    vi_output = []
    for word in translation_graph.src_sent.words:
        word_info = word.info
        word_info["begin"] -= 1
        word_info["end"] -= 1
        word_info.pop("pre")
        word_info.pop("pos")
        word_info.pop("next")
        word_info["ner"] = word_info["ner"] if word_info["ner"] else "O"
        vi_output.append(word_info)

    return vi_output[1:], ba_output[1:]


if __name__ == "__main__":

    executor = ThreadPoolExecutor(max_workers=10)
    # for text in [
    #     "hôm nay là thứ 2",
    #     "mình là Nguyên"
    # ]:
    #     sentence: SyllableBasedSentence = extract_vi_entities(text)
    #     for word in sentence.words:
    #         print(word.info)
    model_checkpoint = "pretrained/best_aligned"

    train_dataset, valid_dataset, test_dataset = ViBaDataset.get_datasets(
        data_folder="data/new_all",
        tokenizer_path=model_checkpoint
    )

    os.makedirs("data/ner/", exist_ok=True)
    for mode, dataset in zip(["train", "valid", "test"], [train_dataset, valid_dataset, test_dataset]):
        ner_data = []


        def job(item):
            vi_sent, ba_sent = item
            vi_entity, ba_entity = data_pipeline(vi_sent=vi_sent, ba_sent=ba_sent)
            return {
                "vi": vi_entity,
                "ba": ba_entity
            }


        for i, _item in enumerate(tqdm(executor.map(job, train_dataset.data), total=len(train_dataset.data))):
            ner_data.append(_item)
            if i % 100 == 0:
                json.dump(
                    ner_data,
                    open(f"data/ner/{mode}_entity.json", "w", encoding="utf8"),
                    ensure_ascii=False,
                    indent=4
                )
        json.dump(
            ner_data,
            open(f"data/ner/{mode}_entity.json", "w", encoding="utf8"),
            ensure_ascii=False,
            indent=4
        )
