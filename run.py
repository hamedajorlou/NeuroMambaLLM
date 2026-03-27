import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from nilearn import datasets, maskers
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 0. REPRODUCIBILITY + CONFIG
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

CACHE_DIR = os.environ.get("HF_HOME", "./hf_cache")
DATA_DIR = os.environ.get("DATA_DIR", "./abide_data")
MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # fallback safe
HF_TOKEN = os.environ.get("HF_TOKEN", None)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# SETTINGS
MAX_SUBJECTS = 50
BATCH_SIZE = 2
EPOCHS = 3
LR = 2e-4
N_ROIS = 39
MAX_SEQ_LEN = 100

SYSTEM_PROMPT = """You are an expert neuroscientist.
Output JSON: {"Diagnosis": "..."}"""

# ==========================================
# 1. MODEL COMPONENTS
# ==========================================
class TemporalGraphEncoder(nn.Module):
    def __init__(self, n_rois, latent_dim=64):
        super().__init__()
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(n_rois, n_rois, 3, padding=1, groups=n_rois),
            nn.ReLU()
        )
        self.time_proj = nn.Linear(MAX_SEQ_LEN, latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.temporal_enc(x)
        return self.time_proj(x), None


class ClinicalBrainLLM(nn.Module):
    def __init__(self, n_rois):
        super().__init__()

        use_4bit = DEVICE == "cuda"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_compute_dtype=DTYPE
        )

        print(f"Loading model: {MODEL_ID}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto" if DEVICE == "cuda" else None,
            quantization_config=bnb_config if use_4bit else None,
            torch_dtype=DTYPE
        )

        if DEVICE != "cuda":
            self.llm.to(DEVICE)

        self.graph = TemporalGraphEncoder(n_rois)
        self.projector = nn.Linear(64, self.llm.config.hidden_size)

    def forward(self, bold, input_ids, labels=None):
        bold = bold.to(DEVICE)

        h, _ = self.graph(bold)
        brain_embeds = self.projector(h)

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([brain_embeds, text_embeds], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, labels=labels)

    def generate(self, bold):
        bold = bold.to(DEVICE)
        h, _ = self.graph(bold)
        brain_embeds = self.projector(h)

        prompt = "Analyze fMRI."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        text_embeds = self.llm.get_input_embeddings()(inputs.input_ids)

        inputs_embeds = torch.cat([brain_embeds, text_embeds], dim=1)

        out = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=20)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# ==========================================
# 2. DATASET
# ==========================================
class ABIDEDataset(Dataset):
    def __init__(self, tokenizer):
        print("Downloading ABIDE (first time may be slow)...")

        atlas = datasets.fetch_atlas_msdl()
        masker = maskers.NiftiMapsMasker(maps_img=atlas.maps, standardize=True)

        abide = datasets.fetch_abide_pcp(
            data_dir=DATA_DIR,
            n_subjects=MAX_SUBJECTS
        )

        self.samples = []
        pheno = pd.DataFrame(abide.phenotypic)

        for i, f in enumerate(abide.func_preproc):
            try:
                ts = masker.fit_transform(f)
                ts = np.nan_to_num(ts)

                if ts.shape[0] < MAX_SEQ_LEN:
                    pad = np.zeros((MAX_SEQ_LEN - ts.shape[0], ts.shape[1]))
                    ts = np.vstack([ts, pad])
                else:
                    ts = ts[:MAX_SEQ_LEN]

                label = pheno.iloc[i]["DX_GROUP"]

                text = json.dumps({
                    "Diagnosis": "ASD" if label == 1 else "Control"
                })

                enc = tokenizer(text, return_tensors="pt", padding="max_length", max_length=64)

                self.samples.append({
                    "bold": torch.tensor(ts, dtype=torch.float32),
                    "input_ids": enc.input_ids.squeeze(),
                    "labels": enc.input_ids.squeeze(),
                    "label": label
                })

            except:
                continue

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i): return self.samples[i]


# ==========================================
# 3. TRAIN + EVAL
# ==========================================
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            bold = batch["bold"]
            text = model.generate(bold)

            pred = 1 if "ASD" in text else 2
            y_pred.append(pred)
            y_true.append(batch["label"].item())

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def main():
    print(f"Running on {DEVICE}")

    model = ClinicalBrainLLM(N_ROIS)
    dataset = ABIDEDataset(model.tokenizer)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            out = model(
                batch["bold"],
                batch["input_ids"],
                batch["labels"]
            )

            loss = out.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
