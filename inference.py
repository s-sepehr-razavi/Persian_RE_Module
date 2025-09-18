import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from TTM.model2 import DocREModel
from hazm import *
from TTM.ner import PersianNER
from dataclasses import dataclass

@dataclass
class TrainArgs:
    data_dir: str
    transformer_type: str
    model_name_or_path: str
    train_file: str
    dev_file: str
    test_file: str
    train_batch_size: int
    test_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_grad_norm: float
    warmup_ratio: float
    num_train_epochs: float
    seed: int
    num_class: int
    isrank: int
    m_tag: str
    model_type: str
    m: float
    e: float
    pretrain_distant: int

# Example usage:
args = TrainArgs(
    data_dir="/kaggle/working",
    transformer_type="bert",
    model_name_or_path="HooshvareLab/bert-fa-base-uncased",
    train_file="train_revised.json",
    dev_file="dev_revised.json",
    test_file="test_revised.json",
    train_batch_size=1,
    test_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,
    max_grad_norm=1.0,
    warmup_ratio=0.06,
    num_train_epochs=30.0,
    seed=74,
    num_class=97,
    isrank=1,
    m_tag="ATLoss",
    model_type="ATLOP",
    m=1.0,
    e=3.0,
    pretrain_distant=0
)



class RelationExtractor:
    def __init__(self, docre_model_name, docre_checkpoint, api_key, num_class=97, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.args = TrainArgs(
            data_dir="/kaggle/working",
            transformer_type="bert",
            model_name_or_path="HooshvareLab/bert-fa-base-uncased",
            train_file="train_revised.json",
            dev_file="dev_revised.json",
            test_file="test_revised.json",
            train_batch_size=1,
            test_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=3e-5,
            max_grad_norm=1.0,
            warmup_ratio=0.06,
            num_train_epochs=30.0,
            seed=74,
            num_class=97,
            isrank=1,
            m_tag="ATLoss",
            model_type="ATLOP",
            m=1.0,
            e=3.0,
            pretrain_distant=0
        )


        # Load NER pipeline
        self.ner_pipeline = PersianNER(api_key)

        # Normalizer
        self.normalizer = Normalizer()

        # DocRE tokenizer (Hazm)
        self.docre_tokenizer = WordTokenizer(join_verb_parts=False)
        
        self.sentence_tokenizer = SentenceTokenizer()

        # Load DocRE model
        config = AutoConfig.from_pretrained(docre_model_name, num_labels=num_class)
        
        backbone = AutoModel.from_pretrained(docre_model_name, config=config).to(self.device)
        priors = torch.ones(num_class).to(self.device) * 1e-9
        self.docre_model = DocREModel(args, config, priors, backbone, self.docre_tokenizer).to(self.device)
        self.model_tokenizer = AutoTokenizer.from_pretrained(docre_model_name)
        self.docre_model.load_state_dict(torch.load(docre_checkpoint, map_location=self.device))
        self.docre_model.eval()

    
    def _compute_offsets_from_tokens(self, text, tokens):
        offsets = []
        search_start = 0
        for tok in tokens:
            start_idx = text.find(tok, search_start)
            if start_idx == -1:
                # print(text)
                raise ValueError(f"Token '{tok}' not found in text starting from {search_start}")
            end_idx = start_idx + len(tok)
            offsets.append((start_idx, end_idx))
            search_start = end_idx
        return offsets


    def _find_all_phrase_token_spans(self, text, phrases):

        tokens = self.tokenizer.tokenize(text)
        offsets = self.compute_offsets_from_tokens(text, tokens)

        results = {}  # { phrase: [(start_token_idx, end_token_idx), ...] }

        for phrase in phrases:
            phrase_spans = []
            search_start = 0    

            while True:
                phrase_start = text.find(phrase, search_start)
                if phrase_start == -1:
                    break
                phrase_end = phrase_start + len(phrase)

                # Find token indices for this occurrence
                token_start = token_end = None
                for idx, (start, end) in enumerate(offsets):
                    if start >= phrase_start and token_start is None:
                        token_start = idx
                    if end <= phrase_end:
                        token_end = idx
                    if start > phrase_end:
                        break

                if token_start is not None and token_end is not None and token_start <= token_end:
                    phrase_spans.append((token_start, token_end))

                # Move search start past this occurrence
                search_start = phrase_start + 1

            results[phrase] = phrase_spans if phrase_spans else [] #BUG

        return tokens, results


    def _tokenize_for_docre(self, text, entities):
        """Tokenize text with Hazm and manually compute offsets"""
        text = self.normalizer.normalize(text)
        sentences = self.sentence_tokenizer.tokenize(text)

        tokens_offsets = []
        for sent in sentences:
            tokens_offsets.append(self._find_all_phrase_token_spans(sent, entities))
        
        mentions_dict = {}
        tokens_list = []
        for sent_idx, tokens, phrase_spans in enumerate(tokens_offsets):
            for entity in entities:
                mentions = []
                for span_start, span_end in phrase_spans[entity]:
                    mention = {
                        "name": entity,
                        "pos": [span_start, span_end],
                        "sent_id": sent_idx,                        
                    }
                if entity in mentions_dict:
                    mentions_dict[entity] += mentions
                else:
                    mentions_dict[entity] = mentions
            
            tokens_list.append(tokens)
        
        entity_mentions = list(mentions_dict.values())

        return tokens_list, entity_mentions

    def _prepare_feature(self, entities, sentences, max_seq_length=1024):          
        sents = []
        sent_map = []

        entity_start, entity_end = [], []        
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]                
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,)) 
        for i_s, sent in enumerate(sentences):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = self.model_tokenizer.tokenize(token)

                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                    
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)



        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))


        hts = []

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t:                                        
                    hts.append([h, t])                    


        sents = sents[:max_seq_length - 2]
        input_ids = self.model_tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.model_tokenizer.build_inputs_with_special_tokens(input_ids)

            
        feature = {'input_ids': input_ids,
                'entity_pos': entity_pos,                
                'hts': hts,            
                }    
        return feature
    
    def predict(self, text):
        # Step 1: Run NER
        entities = self.ner_pipeline.extract_entities(text)
        if not entities:
            return [], [], []

        # Step 2: Tokenize + map entities to token spans                
        
        tokens_list, entity_mentions = self._tokenize_for_docre(text, entities)
        

        feature = self._prepare_feature(entity_mentions, tokens_list)

        # Step 5: Run DocREModel
        with torch.no_grad():
            outputs = self.docre_model(**feature)
        logits = outputs[1].cpu().numpy()

        # Step 6: Convert logits to binary predictions
        preds = np.zeros((logits.shape[0], logits.shape[1] + 1))
        for i in range(logits.shape[1]):
            preds[(logits[:, i] > 0.), i + 1] = 1
        preds[:, 0] = (preds.sum(1) == 0)

        return preds, entities, feature['hts']
