import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification, pipeline
from model2 import DocREModel

class RelationExtractor:
    def __init__(self, ner_model_name, docre_model_name, docre_checkpoint, num_class=97, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load NER pipeline
        self.ner_pipeline = pipeline("ner", model=ner_model_name, aggregation_strategy="simple", device=0 if self.device == "cuda" else -1)

        # DocRE tokenizer
        self.docre_tokenizer = AutoTokenizer.from_pretrained(docre_model_name)

        # Load DocRE model
        config = AutoConfig.from_pretrained(docre_model_name, num_labels=num_class)
        backbone = AutoModel.from_pretrained(docre_model_name, config=config).to(self.device)
        priors = torch.ones(num_class).to(self.device) * 1e-9
        self.docre_model = DocREModel(None, config, priors, backbone, self.docre_tokenizer).to(self.device)
        self.docre_model.load_state_dict(torch.load(docre_checkpoint, map_location=self.device))
        self.docre_model.eval()

    def _tokenize_for_docre(self, text, entities):
        """Tokenize text and map entity char spans to token indices"""
        tokens = self.docre_tokenizer(text, return_offsets_mapping=True, truncation=True)
        offsets = tokens["offset_mapping"]
        entity_pos = []

        for ent in entities:
            start_char, end_char = ent["start"], ent["end"]
            indices = [i for i, (s, e) in enumerate(offsets) if s >= start_char and e <= end_char]
            if indices:
                entity_pos.append(indices)

        return tokens, entity_pos

    def _pair_entities(self, entity_pos):
        """Generate all head-tail index pairs"""
        return [(i, j) for i in range(len(entity_pos)) for j in range(len(entity_pos)) if i != j]

    def predict(self, text):
        # Step 1: Run NER
        entities = self.ner_pipeline(text)
        if not entities:
            return [], [], []

        # Step 2: Tokenize + map entities to token spans
        tokens, entity_pos = self._tokenize_for_docre(text, entities)

        # Step 3: Create entity pairs
        hts = self._pair_entities(entity_pos)
        if not hts:
            return [], entities, []

        # Step 4: Prepare model inputs
        input_ids = torch.tensor([tokens["input_ids"]], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([tokens["attention_mask"]], dtype=torch.float).to(self.device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_pos": [entity_pos],
            "hts": [hts],
        }

        # Step 5: Run DocREModel
        with torch.no_grad():
            outputs = self.docre_model(**inputs)
        logits = outputs[1].cpu().numpy()

        # Step 6: Convert logits to binary predictions
        preds = np.zeros((logits.shape[0], logits.shape[1] + 1))
        for i in range(logits.shape[1]):
            preds[(logits[:, i] > 0.), i + 1] = 1
        preds[:, 0] = (preds.sum(1) == 0)

        return preds, entities, hts
