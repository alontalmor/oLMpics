from typing import Dict, List
import json
import logging
import gzip
import random

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("transformer_mc_qa")
class TransformerMCQAReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 num_choices: int = 4,
                 add_prefix: Dict[str, str] = None,
                 sample: int = -1) -> None:
        super().__init__()

        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model)
        self._tokenizer_internal = self._tokenizer._tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._sample = sample
        self._num_choices = num_choices
        self._add_prefix = add_prefix or {}

        for model in ["roberta", "bert", "openai-gpt", "gpt2", "transfo-xl", "xlnet", "xlm"]:
            if model in pretrained_model:
                self._model_type = model
                break

    @overrides
    def _read(self, file_path: str):
        cached_file_path = cached_path(file_path)

        if file_path.endswith('.gz'):
            data_file = gzip.open(cached_file_path, 'rb')
        else:
            data_file = open(cached_file_path, 'r')


        logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
        item_jsons = []
        for line in data_file:
            item_jsons.append(json.loads(line.strip()))

        if self._sample != -1:
            item_jsons = random.sample(item_jsons, self._sample)
            logger.info("Sampling %d examples", self._sample)

        for item_json in Tqdm.tqdm(item_jsons,total=len(item_jsons)):
            item_id = item_json["id"]

            question_text = item_json["question"]["stem"]

            choice_label_to_id = {}
            choice_text_list = []
            choice_context_list = []
            choice_label_list = []
            choice_annotations_list = []

            any_correct = False
            choice_id_correction = 0

            for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                choice_label = choice_item["label"]
                choice_label_to_id[choice_label] = choice_id - choice_id_correction
                choice_text = choice_item["text"]

                choice_text_list.append(choice_text)
                choice_label_list.append(choice_label)

                if item_json.get('answerKey') == choice_label:
                    if any_correct:
                        raise ValueError("More than one correct answer found for {item_json}!")
                    any_correct = True


            if not any_correct and 'answerKey' in item_json:
                raise ValueError("No correct answer found for {item_json}!")


            answer_id = choice_label_to_id.get(item_json.get("answerKey"))
            # Pad choices with empty strings if not right number
            if len(choice_text_list) != self._num_choices:
                choice_text_list = (choice_text_list + self._num_choices * [''])[:self._num_choices]
                choice_context_list = (choice_context_list + self._num_choices * [None])[:self._num_choices]
                if answer_id is not None and answer_id >= self._num_choices:
                    logging.warning(f"Skipping question with more than {self._num_choices} answers: {item_json}")
                    continue

            yield self.text_to_instance(
                    item_id=item_id,
                    question=question_text,
                    choice_list=choice_text_list,
                    answer_id=answer_id)

        data_file.close()

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice_list: List[str],
                         answer_id: int = None) -> Instance:
        fields: Dict[str, Field] = {}

        qa_fields = []
        segment_ids_fields = []
        qa_tokens_list = []
        annotation_tags_fields = []
        for idx, choice in enumerate(choice_list):
            choice_annotations = []
            qa_tokens, segment_ids = self.transformer_features_from_qa(question, choice)
            qa_field = TextField(qa_tokens, self._token_indexers)
            segment_ids_field = SequenceLabelField(segment_ids, qa_field)
            qa_fields.append(qa_field)
            qa_tokens_list.append(qa_tokens)
            segment_ids_fields.append(segment_ids_field)


        fields['question'] = ListField(qa_fields)
        fields['segment_ids'] = ListField(segment_ids_fields)
        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question,
            "choice_text_list": choice_list,
            "correct_answer_index": answer_id,
            "question_tokens_list": qa_tokens_list,
        }

        if len(annotation_tags_fields) > 0:
            fields['annotation_tags'] = ListField(annotation_tags_fields)
            metadata['annotation_tags'] = [x.array for x in annotation_tags_fields]

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def transformer_features_from_qa(self, question: str, answer: str):
        question = self._add_prefix.get("q", "") + question
        answer = self._add_prefix.get("a",  "") + answer

        # Alon changing mask type:
        if self._model_type in ['roberta','xlnet']:
            question = question.replace('[MASK]','<mask>')
        elif self._model_type in ['albert']:
            question = question.replace('[MASK]', '[MASK]>')

        tokens = self._tokenizer.tokenize_sentence_pair(question, answer)

        # TODO make sure the segments IDs do not contribute
        segment_ids = [0] * len(tokens)

        return tokens, segment_ids