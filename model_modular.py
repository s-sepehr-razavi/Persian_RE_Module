import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributed.pipeline.sync import Pipe



def split_transformer(model: nn.Module, split_layer: int):
    """
    Splits a HuggingFace BERT/Roberta model into two submodules at `split_layer`.
    
    Args:
        model: HuggingFace AutoModel (e.g., BertModel or RobertaModel)
        split_layer: index of encoder layer where to split
    
    Returns:
        part1, part2 (nn.Module)
    """
    base = getattr(model, getattr(model, "base_model_prefix", ""), model)

    # --- Part 1: embeddings + first N layers ---
    part1 = nn.Module()
    part1.embeddings = base.embeddings
    part1.encoder = nn.Module()
    part1.encoder.layer = nn.ModuleList(base.encoder.layer[:split_layer])

    # --- Part 2: remaining layers + pooler (if exists) ---
    part2 = nn.Module()
    part2.encoder = nn.Module()
    part2.encoder.layer = nn.ModuleList(base.encoder.layer[split_layer:])
    part2.pooler = base.pooler if hasattr(base, "pooler") else None

    return part1, part2
# -----------------------------
# Splitter that returns a dict (suitable for nn.Sequential)
# -----------------------------
class LongInputSplitter(nn.Module):
    def __init__(self, max_len=512, start_tokens=None, end_tokens=None):
        super().__init__()
        assert start_tokens is not None and end_tokens is not None
        self.max_len = max_len
        # store as buffers (cpu by default) and cast to input dtype/device at forward time
        self.register_buffer("start_tokens_buf", torch.tensor(start_tokens))
        self.register_buffer("end_tokens_buf", torch.tensor(end_tokens))

    def forward(self, inputs):
        input_ids, attention_mask = inputs
        """
        Accepts: (input_ids, attention_mask)
        Returns: dict with keys:
          - 'chunk_input_ids', 'chunk_attention_mask'   : tensors fed to model
          - 'num_seg', 'seq_len', 'orig_c'             : metadata (lists / ints)
        """
        n, c = input_ids.size()

        # put tokens on same device + dtype as input_ids (mirrors original function)
        start_tokens = self.start_tokens_buf.to(input_ids)
        end_tokens = self.end_tokens_buf.to(input_ids)
        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)

        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()

        # if fits into one chunk, keep original inputs (original function does this path)
        if c <= self.max_len:
            num_seg = [1] * n
            return {
                "chunk_input_ids": input_ids,
                "chunk_attention_mask": attention_mask,
                "num_seg": num_seg,
                "seq_len": seq_len,
                "orig_c": c,
            }

        new_input_ids, new_attention_mask, num_seg = [], [], []
        for i, l_i in enumerate(seq_len):
            if l_i <= self.max_len:
                new_input_ids.append(input_ids[i, :self.max_len])
                new_attention_mask.append(attention_mask[i, :self.max_len])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :self.max_len - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - self.max_len + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :self.max_len]
                attention_mask2 = attention_mask[i, (l_i - self.max_len): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)

        chunk_input_ids = torch.stack(new_input_ids, dim=0)
        chunk_attention_mask = torch.stack(new_attention_mask, dim=0)

        return {
            "chunk_input_ids": chunk_input_ids,
            "chunk_attention_mask": chunk_attention_mask,
            "num_seg": num_seg,
            "seq_len": seq_len,
            "orig_c": c,
        }

# -----------------------------
# ModelWrapper that accepts the dict and stores outputs into it
# -----------------------------
class EncoderPart1(nn.Module):
    def __init__(self, part1, config, model):
        super().__init__()
        self.part1 = part1
        self.config = config
        self.model = model  # needed for get_extended_attention_mask

    def forward(self, data: dict):
        input_ids = data["chunk_input_ids"]
        attention_mask = data["chunk_attention_mask"]
        token_type_ids = data.get("token_type_ids", None)

        # embeddings
        hidden_states = self.part1.embeddings(input_ids=input_ids,
                                              token_type_ids=token_type_ids)

        # build extended attention mask
        extended_attention_mask = self.model.get_extended_attention_mask(
            attention_mask, input_ids.shape, input_ids.device
        )

        # run first half layers
        for layer in self.part1.encoder.layer:
            hidden_states = layer(
                hidden_states,
                extended_attention_mask,
                output_attentions=False
            )[0]

        # update dict
        data["hidden_states_part1"] = hidden_states
        data["extended_attention_mask"] = extended_attention_mask
        return data


class EncoderPart2(nn.Module):
    def __init__(self, part2, config):
        super().__init__()
        self.part2 = part2
        self.config = config

    def forward(self, data: dict):
        hidden_states = data["hidden_states_part1"]
        extended_attention_mask = data["extended_attention_mask"]

        attentions_all = []
        for layer in self.part2.encoder.layer:
            layer_outputs = layer(
                hidden_states,
                extended_attention_mask,
                output_attentions=True
            )
            hidden_states = layer_outputs[0]
            attentions_all.append(layer_outputs[1])

        # update dict
        data["sequence_output"] = hidden_states
        data["attention"] = attentions_all[-1]   # last-layer attention
        data["attentions"] = attentions_all      # all attentions in part2

        if self.part2.pooler is not None:
            data["pooled_output"] = self.part2.pooler(hidden_states)

        return data

# -----------------------------
# Recombiner that consumes the dict and returns (sequence_output, attention)
# -----------------------------
class LongInputRecombiner(nn.Module):
    def __init__(self, max_len=512, start_tokens=None, end_tokens=None):
        super().__init__()
        assert start_tokens is not None and end_tokens is not None
        self.max_len = max_len
        self.len_start = len(start_tokens)
        self.len_end = len(end_tokens)

    def forward(self, data: dict):
        sequence_output = data["sequence_output"]
        attention = data["attention"]
        chunk_attention_mask = data["chunk_attention_mask"]
        num_seg = data["num_seg"]
        seq_len = data["seq_len"]
        c = data["orig_c"]

        # If no splitting occurred, the original function returned model outputs directly.
        if c <= self.max_len:
            return sequence_output, attention

        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                out = F.pad(sequence_output[i], (0, 0, 0, c - self.max_len))
                att = F.pad(attention[i], (0, c - self.max_len, 0, c - self.max_len))
                new_output.append(out)
                new_attention.append(att)
            elif n_s == 2:
                out1 = sequence_output[i][:self.max_len - self.len_end]
                m1 = chunk_attention_mask[i][:self.max_len - self.len_end]
                att1 = attention[i][:, :self.max_len - self.len_end, :self.max_len - self.len_end]
                out1 = F.pad(out1, (0, 0, 0, c - self.max_len + self.len_end))
                m1 = F.pad(m1, (0, c - self.max_len + self.len_end))
                att1 = F.pad(att1, (0, c - self.max_len + self.len_end, 0, c - self.max_len + self.len_end))

                out2 = sequence_output[i + 1][self.len_start:]
                m2 = chunk_attention_mask[i + 1][self.len_start:]
                att2 = attention[i + 1][:, self.len_start:, self.len_start:]
                out2 = F.pad(out2, (0, 0, l_i - self.max_len + self.len_start, c - l_i))
                m2 = F.pad(m2, (l_i - self.max_len + self.len_start, c - l_i))
                att2 = F.pad(att2, [l_i - self.max_len + self.len_start, c - l_i,
                                    l_i - self.max_len + self.len_start, c - l_i])

                mask = m1 + m2 + 1e-10
                out = (out1 + out2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)

                new_output.append(out)
                new_attention.append(att)
            i += n_s

        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
        return sequence_output, attention


class HRTExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # BERT and RoBERTa add a [CLS] token at the start → offset=1
        self.offset = 1 if config.transformer_type in ["bert", "roberta"] else 0
        self.hidden_size = config.hidden_size

    def forward(self, data):
        

        sequence_output, attention, entity_pos, hts = data['sequence_output'], data['attention'], data['entity_pos'], data['hts']
        """
        Args:
            sequence_output: Tensor (n, L, d) - encoder hidden states
            attention: Tensor (n, h, L, L) - attention weights
            entity_pos: list[list[list[tuple]]] - entity mentions per sample
            hts: list[list[tuple]] - head-tail index pairs per sample

        Returns:
            hss: Tensor (N_rel, d)
            rss: Tensor (N_rel, d)
            tss: Tensor (N_rel, d)
        """
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []

        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []

            # Encode each entity
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + self.offset < c:  # skip if truncated
                            e_emb.append(sequence_output[i, start + self.offset])
                            e_att.append(attention[i, :, start + self.offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.hidden_size, device=sequence_output.device)
                        e_att = torch.zeros(h, c, device=attention.device)
                else:
                    start, end = e[0]
                    if start + self.offset < c:
                        e_emb = sequence_output[i, start + self.offset]
                        e_att = attention[i, :, start + self.offset]
                    else:
                        e_emb = torch.zeros(self.hidden_size, device=sequence_output.device)
                        e_att = torch.zeros(h, c, device=attention.device)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)

            # If no relations in this sample
            if len(hts[i]) == 0:
                hss.append(torch.empty(0, self.hidden_size, device=sequence_output.device))
                tss.append(torch.empty(0, self.hidden_size, device=sequence_output.device))
                rss.append(torch.empty(0, self.hidden_size, device=sequence_output.device))
                continue

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)

            rs = torch.einsum("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)

        data['hs'] = hss
        data['ts'] = tss
        data['rs'] = rss

        return data


class ATLOPLayer(nn.Module):
    def __init__(self, args, hidden_size, emb_size, block_size, num_labels):
        """
        Args:
            args: namespace or config object containing at least `model_type` and `m_tag`
            hidden_size: the same thing as config.hidden_size in original implementation
            emb_size: embedding size after concat (hs+rs or ts+rs)
            block_size: block factor for bilinear pooling
            num_labels: number of relation classes
        """
        super().__init__()
        self.args = args
        self.emb_size = emb_size
        self.block_size = block_size

        # Extractors (map [hs+rs] or [ts+rs] → emb_size)
        self.head_extractor = nn.Linear(hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(hidden_size * 2, emb_size)

        # Bilinear classifier
        self.bilinear = nn.Linear(emb_size * block_size, num_labels)

    def forward(self, data):

        hs, ts, rs = data['hs'], data['ts'], data['rs']
        """
        Args:
            hs: Tensor (N, d) - head embeddings
            ts: Tensor (N, d) - tail embeddings
            rs: Tensor (N, d) - relation-aware embeddings

        Returns:
            logits_list: [Tensor(N, num_labels)]
            loss_weights_list: [float]
            m_tags_list: [str]
        """
        if self.args.model_type != "ATLOP":
            raise ValueError(f"ATLOPLayer only supports ATLOP, got {self.args.model_type}")

        # Head/tail extractors
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))  # zs
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))  # zo

        # Block bilinear interaction
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)

        logits = self.bilinear(bl)

        logits_list = [logits]
        loss_weights_list = [1.0]
        m_tags_list = [self.args.m_tag]

        return logits_list, loss_weights_list, m_tags_list


class PartitionedDocREModel(nn.Module):
    def __init__(self, args, config, priors_l, model, tokenizer, devices, emb_size=768, block_size=64):
        super().__init__()
        self.args = args
        self.config = config                
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size
        self.priors_l = priors_l
        self.priors_o = priors_l * args.e
        self.priors_u = (self.priors_o - self.priors_l) / (1. - self.priors_l)
        self.weight = ((1 - self.priors_o)/self.priors_o) ** 0.5
        self.margin = args.m
        self.devices = devices
        self.retrieval_weight = .2
        self.train_mode = 'finetune' # 'finetune' or 'pretrain' 
        if args.isrank:
            self.rels = args.num_class-1
        else:
            self.rels = args.num_class
        self.emb_size = emb_size
        self.block_size = block_size
        self.model = self.preparing_partitioned_model(model)
        
    
    
    def preparing_partitioned_model(self, model):
        
        if self.config.transformer_type == "bert":
            start_tokens = [self.config.cls_token_id]
            end_tokens = [self.config.sep_token_id]
        elif self.config.transformer_type == "roberta":
            start_tokens = [self.config.cls_token_id]
            end_tokens = [self.config.sep_token_id, self.config.sep_token_id]
        
        splitter = LongInputSplitter(512, start_tokens, end_tokens)
        
        
        part1, part2 = split_transformer(model, split_layer=6) # beware of the number you set for the split_layer param

        part1 = EncoderPart1(part1)
        part2 = EncoderPart2(part2)

        recombiner = LongInputRecombiner(512, start_tokens, end_tokens)

        hrtextractor = HRTExtractor(self.config) 

        atlop = ATLOPLayer(self.args, 2 * self.config.hidden_size, self.emb_size, self.block_size, self.config.num_labels)

        model = nn.Sequential(
            nn.Sequential(
            splitter,
            part1).to(self.devices[0]),
            nn.Sequential(            
            part2,
            recombiner,
            hrtextractor,
            atlop
            ).to(self.devices[1])
        )

        model = Pipe(model, chunks=2)
        return model
    
    def square_loss(self, yPred, yTrue, margin=1.):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = (yPred * yTrue - margin) ** 2
        return torch.mean(loss.sum() / yPred.shape[0])

    def forward(self, input_ids, attention_mask, token_type_ids=None, entity_pos=None, hts=None, labels=None):
        """
        Args:
            input_ids: Tensor (n, L)
            attention_mask: Tensor (n, L)
            token_type_ids: Tensor (n, L) or None
            entity_pos: list[list[list[tuple]]] - entity mentions per sample
            hts: list[list[tuple]] - head-tail index pairs per sample
            labels: Tensor (N_rel,) or None - target relation labels

        Returns:
            if labels is provided:
                loss: Tensor (1,)
                logits_list: [Tensor(N_rel, num_labels)]
                loss_weights_list: [float]
                m_tags_list: [str]
            else:
                logits_list: [Tensor(N_rel, num_labels)]
                loss_weights_list: [float]
                m_tags_list: [str]
        """
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            data["token_type_ids"] = token_type_ids
        data["entity_pos"] = entity_pos
        data["hts"] = hts
                
        

        data = self.model(data).local_value()  # get output from last stage of Pipe

        logits_list, loss_weights_list, m_tags_list = data

        risk_sum = []
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(self.devices[-1])

            for logits, loss_weight, m_tag in zip(logits_list, loss_weights_list, m_tags_list):
                # Partition
                if m_tag == 'ATLoss':
                    assert self.args.isrank == True
                    """https://github.com/YoumiMa/dreeam/blob/main/losses.py"""
                    labels = labels.clone()
                    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
                    th_label[:, 0] = 1.0
                    labels[:, 0] = 0.0
                    # Rank positive classes highly
                    logit1 = logits - (1 - labels - th_label) * 1e30
                    loss1 = -(nn.functional.log_softmax(logit1, dim=-1) * labels).sum(1)
                    # Rank negative classes lowly
                    logit2 = logits - labels * 1e30
                    loss2 = -(nn.functional.log_softmax(logit2, dim=-1) * th_label).sum(1)
                    # Sum two parts
                    loss = loss1 + loss2
                    risk_sum.append(loss.mean() * loss_weight)                
            
            return risk_sum, logits_list[0]
        


        return logits_list[0]

