import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from nilearn import datasets, maskers
import numpy as np
import pandas as pd
import json
import re
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 0. CONFIG
# ==========================================
USER = "prazmara"
CACHE_DIR = f"/scratch1/{USER}/hf_cache"
DATA_DIR = f"/scratch1/{USER}/abide_data"
CKPT_PATH = f"/scratch1/{USER}/neuromamba_v7_debug.pt"

os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- SETTINGS ---
MAX_SUBJECTS = 100   # Limit to 100 for fast debugging
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 15
LR = 2e-4
N_ROIS = 39
MAX_SEQ_LEN = 100

# ==========================================
# 1. SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """You are an expert neuroscientist. Analyze the provided brain connectivity embedding.
Output valid JSON only.
Format: {"Diagnosis": "Autism Spectrum Disorder (ASD)"} or {"Diagnosis": "Typically Developing Control (TC)"}.
"""

# ==========================================
# 2. ARCHITECTURE
# ==========================================
class TemporalGraphEncoder(nn.Module):
    def __init__(self, n_rois, latent_dim=64):
        super().__init__()
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(n_rois, n_rois * 2, kernel_size=3, padding=1, groups=n_rois),
            nn.BatchNorm1d(n_rois * 2), nn.ReLU(),
            nn.Conv1d(n_rois * 2, n_rois, kernel_size=3, padding=1, groups=n_rois),
            nn.BatchNorm1d(n_rois), nn.ReLU()
        )
        self.time_proj = nn.Linear(MAX_SEQ_LEN, latent_dim)
        self.W_q = nn.Linear(latent_dim, latent_dim)
        self.W_k = nn.Linear(latent_dim, latent_dim)
        self.scale = latent_dim ** -0.5

    def forward(self, x):
        x = x.transpose(1, 2) # [B, N, T]
        x_temp = self.temporal_enc(x)
        h = self.time_proj(x_temp)
        Q = self.W_q(h)
        K = self.W_k(h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        adj = F.softmax(scores, dim=-1)
        return h, adj

class ClinicalBrainLLM(nn.Module):
    def __init__(self, n_rois):
        super().__init__()
        self.llm_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"Loading {self.llm_id}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_id, token=os.environ.get("HF_TOKEN"))
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)

        d_latent = 128
        self.graph_module = TemporalGraphEncoder(n_rois, latent_dim=128)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_latent, nhead=4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.projector = nn.Linear(d_latent, self.llm.config.hidden_size)
        self.ln_llm = nn.LayerNorm(self.llm.config.hidden_size)
        self.query_tokens = nn.Parameter(torch.randn(1, 8, d_latent))

        # FORCE BFLOAT16 & CUDA
        device = "cuda"
        dtype = torch.bfloat16
        self.graph_module.to(device=device, dtype=dtype)
        self.encoder.to(device=device, dtype=dtype)
        self.projector.to(device=device, dtype=dtype)
        self.ln_llm.to(device=device, dtype=dtype)
        self.query_tokens.data = self.query_tokens.data.to(device=device, dtype=dtype)

    def forward(self, bold, input_ids, attention_mask=None, labels=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bold = torch.nan_to_num(bold, nan=0.0).to(self.llm.device)
            h, adj = self.graph_module(bold)
            h_graph = torch.matmul(adj, h)
            feat = self.encoder(h_graph)
            q = self.query_tokens.repeat(bold.shape[0], 1, 1)
            context = F.scaled_dot_product_attention(q, feat, feat)
            brain_embeds = self.projector(context)
            brain_embeds = self.ln_llm(brain_embeds)
            text_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([brain_embeds, text_embeds], dim=1)

            if attention_mask is not None:
                brain_mask = torch.ones((bold.shape[0], 8), device=bold.device)
                attention_mask = torch.cat([brain_mask, attention_mask], dim=1)

            if labels is not None:
                brain_labels = torch.full((bold.shape[0], 8), -100, device=bold.device)
                labels = torch.cat([brain_labels, labels], dim=1)

            outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

    def generate_report(self, bold):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bold = torch.nan_to_num(bold, nan=0.0).to(self.llm.device)
            h, adj = self.graph_module(bold)
            h_graph = torch.matmul(adj, h)
            feat = self.encoder(h_graph)
            q = self.query_tokens.repeat(bold.shape[0], 1, 1)
            context = F.scaled_dot_product_attention(q, feat, feat)
            brain_embeds = self.projector(context)
            brain_embeds = self.ln_llm(brain_embeds)

            # Inference Prompt
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAnalyze the fMRI scan.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            text_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
            inputs_embeds = torch.cat([brain_embeds, text_embeds], dim=1)

            # Mask
            combined_mask = torch.ones(inputs_embeds.shape[:2], device=self.llm.device)

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_mask,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==========================================
# 3. DATASET
# ==========================================
class InstructionABIDEDataset(Dataset):
    def __init__(self, tokenizer, data_dir=DATA_DIR):
        self.tokenizer = tokenizer
        self.atlas = datasets.fetch_atlas_msdl()
        self.masker = maskers.NiftiMapsMasker(
            maps_img=self.atlas.maps, standardize="zscore_sample",
            memory=os.path.join(data_dir, 'nilearn_cache'), verbose=0
        )
        print("Loading Dataset...")
        self.abide = datasets.fetch_abide_pcp(data_dir=data_dir, n_subjects=None, pipeline="cpac", quality_checked=True)
        self.data = []
        self._process_data()

    def _process_data(self):
        pheno = pd.DataFrame(self.abide.phenotypic)
        count = 0
        for i, func_file in enumerate(self.abide.func_preproc):
            if MAX_SUBJECTS and count >= MAX_SUBJECTS: break
            try:
                if not os.path.exists(func_file): continue
                ts = self.masker.fit_transform(func_file)
                if np.isnan(ts).any(): ts = np.nan_to_num(ts, nan=0.0)

                T, N = ts.shape
                if T > MAX_SEQ_LEN: ts = ts[:MAX_SEQ_LEN, :]
                elif T < MAX_SEQ_LEN:
                    padding = np.zeros((MAX_SEQ_LEN - T, N))
                    ts = np.vstack([ts, padding])

                dx_raw = pheno.iloc[i]['DX_GROUP']
                dx = "Autism Spectrum Disorder (ASD)" if dx_raw == 1 else "Typically Developing Control (TC)"

                clinical_json = json.dumps({"Diagnosis": dx})

                self.data.append({
                    'bold': torch.tensor(ts, dtype=torch.float32),
                    'json': clinical_json,
                    'label': dx_raw # 1=ASD, 2=Control
                })
                count += 1
                if count % 10 == 0: print(f"  Loaded {count}...")
            except: pass
        print(f"Total Subjects: {len(self.data)}")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]

        user_input = "Analyze the fMRI scan."
        full_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{item['json']}<|eot_id|>"
        )

        encoded = self.tokenizer(full_text, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        labels = input_ids.clone()

        prompt_len = len(self.tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n").input_ids)
        labels[:prompt_len] = -100

        return {
            'bold': item['bold'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'raw_label': item['label']
        }

# ==========================================
# 4. PIPELINE (TRAIN + EVAL)
# ==========================================
def evaluate_model(model, dataloader):
    print("\n--- Starting Evaluation on Test Set ---")
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # --- FIX: REMOVE UNSQUEEZE TO PREVENT DOUBLE BATCHING ---
            bold = batch['bold'].to(model.llm.device) # Shape is already [1, 100, 39]

            true_label = batch['raw_label'].item()

            # Generate Report
            text_out = model.generate_report(bold)

            # Parse Output
            pred_label = 2 # Default to Control
            if "Autism" in text_out or "ASD" in text_out:
                pred_label = 1

            y_true.append(true_label)
            y_pred.append(pred_label)

            if i < 3: print(f"Sample {i}: True={true_label}, Pred={pred_label} | Text: {text_out[:50]}...")

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== FINAL ACCURACY: {acc*100:.2f}% ===")
    print(classification_report(y_true, y_pred, target_names=["Autism (1)", "Control (2)"], zero_division=0))

def run_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running V7 (Fast Debug - Limit 100) on {device}")

    model = ClinicalBrainLLM(n_rois=N_ROIS)
    dataset = InstructionABIDEDataset(model.tokenizer)

    # SPLIT DATA (80% Train, 20% Test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Data Split: {len(train_dataset)} Train, {len(test_dataset)} Test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 for inference

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n--- Training ({EPOCHS} Epochs) ---")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            bold = batch['bold'].to(model.llm.device)
            input_ids = batch['input_ids'].to(model.llm.device)
            mask = batch['attention_mask'].to(model.llm.device)
            labels = batch['labels'].to(model.llm.device)

            outputs = model(bold, input_ids, mask, labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS

        scheduler.step()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # RUN EVALUATION
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    run_pipeline()
