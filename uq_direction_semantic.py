import re, gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import networkx as nx
from scipy import special


# -----------------------------
# Utility
# -----------------------------
def normalize_generation(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------
# vMF estimation 
# -----------------------------
def _A_p_stable(kappa, d):
    if kappa <= 1e-8:
        return kappa / d
    if kappa >= 1e4:
        return 1.0 - (d-1)/(2.0*kappa) + (d-1)*(d-3)/(4.0*kappa**2)
    num = special.ive(d/2.0, kappa)
    den = special.ive(d/2.0 - 1.0, kappa)
    return num / (den + 1e-300)

def _dA_p_from_A(kappa, A, d):
    if kappa <= 0:
        return 1.0 - A*A
    return 1.0 - A*A - ((d-1)/kappa)*A

def _kappa_initial(Rbar, d):
    Rbar = float(np.clip(Rbar, 1e-12, 1 - 1e-12))
    return max((Rbar * (d - Rbar**2)) / (1.0 - Rbar**2), 1e-6)

def _kappa_upper_bound(Rbar, d):
    Rbar = float(np.clip(Rbar, 1e-9, 1 - 1e-9))
    return max((d-1) / (2.0 * (1.0 - Rbar)), 10.0)

def vmf_stats(embeddings, newton_iters=8, tol=1e-7, use_bisection_backup=True):
    X = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    N, d = X.shape
    R = X.sum(axis=0)
    R_norm = float(np.linalg.norm(R))
    if R_norm < 1e-12:
        return {"kappa": 0.0, "trace_var": 1.0, "var_parallel": 1.0/d, "var_perp": 1.0/d}

    Rbar = R_norm / max(N, 1)

    kappa = _kappa_initial(Rbar, d)
    ok_newton = True
    for _ in range(newton_iters):
        A = _A_p_stable(kappa, d)
        f = A - Rbar
        if abs(f) < tol:
            break
        dA = _dA_p_from_A(kappa, A, d)
        step = f / (dA + 1e-18)
        kappa_new = kappa - step
        if (not np.isfinite(kappa_new)) or (kappa_new <= 0) or (abs(step) > 0.5*max(1.0, kappa)):
            ok_newton = False
            break
        kappa = kappa_new

    if (not ok_newton) and use_bisection_backup:
        lo = 0.0
        hi = _kappa_upper_bound(Rbar, d)
        for _ in range(60):
            mid = 0.5*(lo + hi)
            A_mid = _A_p_stable(mid, d)
            if A_mid < Rbar:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < 1e-7 * (1.0 + mid):
                break
        kappa = 0.5*(lo + hi)

    A = float(Rbar)
    var_perp = float(max(A / (kappa + 1e-18), 0.0))
    var_parallel = float(max(1.0 - ((d - 1.0) * A / (kappa + 1e-18)) - A*A, 0.0))
    trace_var = float(max(1.0 - A*A, 0.0))

    return {"kappa": float(kappa), "trace_var": trace_var, "var_parallel": var_parallel, "var_perp": var_perp}


# -----------------------------
#  Semantic entropy scorer
# -----------------------------
class DebertaSemanticEntropy:
    def __init__(self, cache_dir: str, device: torch.device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli", cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-large-mnli", cache_dir=cache_dir
        ).to(device)
        self.model.eval()

    def _entail_prob(self, premise: str, hypothesis: str) -> float:
        inputs = self.tok(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        return probs[0][2].item()

    def score(self, gens, threshold=0.8) -> float:
        n = len(gens)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1, n):
                e1 = self._entail_prob(gens[i], gens[j])
                e2 = self._entail_prob(gens[j], gens[i])
                if e1 > threshold and e2 > threshold:
                    G.add_edge(i, j)
        comps = list(nx.connected_components(G))
        labels = np.zeros(n, dtype=int)
        for cid, nodes in enumerate(comps):
            for node in nodes:
                labels[node] = cid
        counts = np.bincount(labels)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p + 1e-10)))


# -----------------------------
# Model loading + generation
# -----------------------------
def load_llm(model_id: str, hf_token: str, cache_dir: str, device, dtype):
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, cache_dir=cache_dir, torch_dtype=dtype
    ).to(device)
    model.eval()
    return tok, model

def generate_completions(tok, model, prompt: str, *, n: int, batch_size: int,
                         max_new_tokens: int, temperature: float, top_p: float, device):
    gens = []
    inputs = tok(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids.repeat(bs, 1),
                attention_mask=attention_mask.repeat(bs, 1),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=bs,
                pad_token_id=tok.pad_token_id,
            )
        for i in range(bs):
            gen_ids = out[i][input_ids.shape[1]:]
            gens.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
        clear_gpu()

    return gens


# -----------------------------
# Prompt builders (general)
# -----------------------------
def make_prompt_from_template(template: str):
    """
    template can use df column names, e.g.
      "Q: {question}\nContext: {context}\nA:"
    """
    def _fn(row: pd.Series) -> str:
        return template.format(**row.to_dict())
    return _fn


# -----------------------------
# Main runner (DATASET-AGNOSTIC)
# -----------------------------
def run_uq(
    df: pd.DataFrame,
    model_id: str,
    *,
    prompt_fn=None,                 # function(row)->str
    prompt_template: str = None,    # OR a string template
    out_csv: str = None,
    hf_token: str = "",
    cache_dir: str = "",
    embed_model_name: str = "intfloat/e5-large-v2",
    compute_semantic_entropy: bool = True,
    entail_threshold: float = 0.8,
    n_generations: int = 10,
    batch_size: int = 2,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    if (prompt_fn is None) == (prompt_template is None):
        raise ValueError("Provide exactly one of: prompt_fn or prompt_template")

    if prompt_template is not None:
        prompt_fn = make_prompt_from_template(prompt_template)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tok, llm = load_llm(model_id, hf_token, cache_dir, device, dtype)
    embedder = SentenceTransformer(embed_model_name, device="cpu")

    se = DebertaSemanticEntropy(cache_dir=cache_dir, device=device) if compute_semantic_entropy else None

    rows = []
    for i, row in tqdm(df.reset_index(drop=True).iterrows(), total=len(df)):
        prompt = prompt_fn(row)
        gens = generate_completions(
            tok, llm, prompt,
            n=n_generations, batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            device=device
        )

        cleaned = [f"passage: {normalize_generation(g)}" for g in gens if isinstance(g, str) and g.strip()]
        if len(cleaned) == 0:
            stats = {"kappa": np.nan, "trace_var": np.nan, "var_parallel": np.nan, "var_perp": np.nan}
        else:
            emb = embedder.encode(cleaned, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
            stats = vmf_stats(emb)

        se_val = np.nan
        if se is not None:
            try:
                se_val = se.score(gens, threshold=entail_threshold)
            except Exception:
                se_val = np.nan

        # store everything; no required column names
        out_row = {"model_id": model_id, "row_id": i, "generations": gens, **stats, "semantic_entropy": se_val}
        for col in df.columns:
            out_row[col] = row[col]
        rows.append(out_row)

    out = pd.DataFrame(rows)
    if out_csv:
        out.to_csv(out_csv, index=False)

    del llm, tok
    clear_gpu()
    return out
