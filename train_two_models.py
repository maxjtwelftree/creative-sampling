"""
1. base model (smaller GPT‑2) is trained on *first half* of the dataset.
2. the trained base model scores tokens (ranks) on the *second half*; a histogram of these ranks is saved.
3. ranks are quantised `rank_id` buckets.
4. meta model (larger GPT‑2 with extra *rank embeddings*) is trained on the ranked second‑half data.
"""
import os
import sys
import pickle
import traceback
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)

def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    import psutil
    print(f"RAM: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.1f} MB")

class IrishFolkDataset(Dataset):
    """Returns dict with *input_ids* and *labels* (causal LM)"""
    def __init__(self, seqs: np.ndarray):
        self.seqs = seqs.astype(np.int64)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor(self.seqs[idx], dtype=torch.long)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask, "labels": ids}

class RankedIrishFolkDataset(Dataset):
    """Dataset with *music token* & *rank bucket token* for each position."""
    def __init__(self, seqs: np.ndarray, rank_ids: np.ndarray):
        assert seqs.shape == rank_ids.shape
        self.seqs = seqs.astype(np.int64)
        self.rank_ids = rank_ids.astype(np.int64)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor(self.seqs[idx], dtype=torch.long)
        rank_ids = torch.tensor(self.rank_ids[idx], dtype=torch.long)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "rank_ids": rank_ids, "attention_mask": mask, "labels": ids}

def token_ranks_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Return rank (0 = most likely) of *target_ids* within *logits* distribution."""
    ranks = torch.argsort(torch.argsort(-logits, dim=-1), dim=-1)
    target_ranks = ranks.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return target_ranks  # shape: (seq_len)

@torch.no_grad()
def generate_rank_matrix(model, data_loader, device) -> np.ndarray:
    model.eval()
    rank_list: List[np.ndarray] = []
    for batch in data_loader:
        ids = batch["input_ids"].to(device)
        outputs = model(ids).logits  # (B, L, V)
        for i in range(ids.size(0)):
            ranks = token_ranks_from_logits(outputs[i], ids[i])
            rank_list.append(ranks.cpu().numpy())
    return np.stack(rank_list)

class GPT2WithRank(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config, n_rank_buckets: int, rank_emb_dim: int = 64):
        super().__init__(config)
        self.rank_embeddings = torch.nn.Embedding(n_rank_buckets, rank_emb_dim)
        # Project rank_emb_dim to model dim and add to token embeddings
        self.rank_proj = torch.nn.Linear(rank_emb_dim, config.n_embd, bias=False)
        self.init_weights()

    def forward(self, input_ids=None, rank_ids=None, **kwargs):
        if rank_ids is None:
            raise ValueError("rank_ids required for meta model")
        # Token embeddings from GPT‑2
        inputs_embeds = self.transformer.wte(input_ids)
        # Rank embeds
        rank_embeds = self.rank_proj(self.rank_embeddings(rank_ids))
        inputs_embeds = inputs_embeds + rank_embeds  # simple additive fusion
        return super().forward(inputs_embeds=inputs_embeds, **kwargs)

def main():
    processed_dir = "./data/processed"
    output_root = "./models/third/huggingface-gpt2"
    os.makedirs(output_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device → {device}")

    seqs = np.load(os.path.join(processed_dir, "inputs.npy"))  # (N, seq_len)
    with open(os.path.join(processed_dir, "vocab_size.txt")) as f:
        vocab_size = int(f.read())
    print(f"Loaded {len(seqs):,} sequences · vocab={vocab_size}")
    mid = len(seqs) // 2
    seqs_base, seqs_meta = seqs[:mid], seqs[mid:]

    tokenizer = PreTrainedTokenizerFast(
        bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]",
        additional_special_tokens=[f"[NOTE_{i}]" for i in range(vocab_size)] + ["[SEP]", "[REST]", "[TIME_SHIFT]"]
    )

    base_cfg = GPT2Config(vocab_size=vocab_size + 10, n_positions=128, n_embd=512, n_layer=8, n_head=8, bos_token_id=vocab_size, eos_token_id=vocab_size + 1, pad_token_id=vocab_size + 2)
    base_model = GPT2LMHeadModel(base_cfg).to(device)

    train_base_ds = IrishFolkDataset(seqs_base)
    base_args = TrainingArguments(output_dir=os.path.join(output_root, "base"), num_train_epochs=20, per_device_train_batch_size=8, gradient_accumulation_steps=2, fp16=torch.cuda.is_available(), evaluation_strategy="no", save_strategy="epoch", report_to="none")
    base_trainer = Trainer(model=base_model, args=base_args, train_dataset=train_base_ds,data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    print("Training base model…")
    base_trainer.train()
    base_trainer.save_model(os.path.join(output_root, "base_final"))

    print("Scoring second‑half with base model…")
    meta_ds_plain = IrishFolkDataset(seqs_meta)
    meta_loader = torch.utils.data.DataLoader(meta_ds_plain, batch_size=8, shuffle=False)
    rank_matrix = generate_rank_matrix(base_model, meta_loader, device)  # (M, seq_len)

    import matplotlib.pyplot as plt
    flat_ranks = rank_matrix.flatten()
    plt.hist(flat_ranks, bins=100)
    hist_path = os.path.join(output_root, "rank_hist.png")
    plt.savefig(hist_path)
    print(f"Histogram saved → {hist_path}")

    quantiles = np.quantile(flat_ranks, [0.25, 0.5, 0.75])
    def bucketize(r):
        return 0 if r <= quantiles[0] else 1 if r <= quantiles[1] else 2 if r <= quantiles[2] else 3
    bucket_vec = np.vectorize(bucketize)(rank_matrix).astype(np.int64)
    n_buckets = 4

    # ---------- Meta model ---------- #
    meta_cfg = GPT2Config(vocab_size=vocab_size + 10, n_positions=128, n_embd=768, n_layer=12, n_head=12,
                          bos_token_id=vocab_size, eos_token_id=vocab_size + 1, pad_token_id=vocab_size + 2)
    meta_model = GPT2WithRank(meta_cfg, n_rank_buckets=n_buckets).to(device)

    meta_dataset = RankedIrishFolkDataset(seqs_meta, bucket_vec)
    meta_args = TrainingArguments(output_dir=os.path.join(output_root, "meta"), num_train_epochs=30,
                                  per_device_train_batch_size=8, gradient_accumulation_steps=2, fp16=torch.cuda.is_available(),
                                  evaluation_strategy="no", save_strategy="epoch", report_to="none")
    data_collator_meta = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    class RankDataCollator:
        def __init__(self, base_collator):
            self.base_collator = base_collator
        def __call__(self, features):
            batch = self.base_collator([{k: f[k] for k in ["input_ids", "attention_mask", "labels"]} for f in features])
            batch["rank_ids"] = torch.stack([f["rank_ids"] for f in features])
            return batch

    meta_trainer = Trainer(model=meta_model, args=meta_args, train_dataset=meta_dataset,
                           data_collator=RankDataCollator(data_collator_meta))
    print("Training meta model…")
    meta_trainer.train()
    meta_trainer.save_model(os.path.join(output_root, "meta_final"))
    print("✅ Pipeline complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Unhandled exception:", e)
        traceback.print_exc()
        sys.exit(1)
