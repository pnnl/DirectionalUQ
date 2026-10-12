import json
import re
from typing import List, Dict, Any, Optional, Union

import numpy as np
from scipy import special
from sentence_transformers import SentenceTransformer


# -----------------------------
# Text cleaning
# -----------------------------
def normalize_text(text: str) -> str:
    return text.strip()


# -----------------------------
# Stable vMF helpers
# -----------------------------
def _A_p_stable(kappa: float, d: int) -> float:
    if kappa <= 1e-8:
        return kappa / d

    if kappa >= 1e4:
        return (
            1.0
            - (d - 1) / (2.0 * kappa)
            + (d - 1) * (d - 3) / (4.0 * kappa**2)
        )

    num = special.ive(d / 2.0, kappa)
    den = special.ive(d / 2.0 - 1.0, kappa)

    return float(num / (den + 1e-300))


def _dA_p_from_A(kappa: float, A: float, d: int) -> float:
    if kappa <= 0:
        return 1.0 - A * A

    return 1.0 - A * A - ((d - 1) / kappa) * A


def _kappa_initial(Rbar: float, d: int) -> float:
    Rbar = float(np.clip(Rbar, 1e-12, 1 - 1e-12))
    return max((Rbar * (d - Rbar**2)) / (1.0 - Rbar**2), 1e-6)


def _kappa_upper_bound(Rbar: float, d: int) -> float:
    Rbar = float(np.clip(Rbar, 1e-9, 1 - 1e-9))
    return max((d - 1) / (2.0 * (1.0 - Rbar)), 10.0)


# -----------------------------
# Core vMF / DCU computation
# -----------------------------
def compute_vmf_stats(
    embeddings: np.ndarray,
    newton_iters: int = 8,
    tol: float = 1e-7,
) -> Dict[str, float]:

    X = np.asarray(embeddings, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("embeddings must be a 2D array of shape (n_samples, embedding_dim).")

    n, d = X.shape

    if n == 0:
        raise ValueError("At least one embedding is required.")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-12, None)

    R = X.sum(axis=0)
    R_norm = float(np.linalg.norm(R))

    if R_norm < 1e-12:
        return {
            "kappa": 0.0,
            "dcu": float("inf"),
            "trace_var": 1.0,
            "mean_resultant_length": 0.0,
            "n_generations": n,
            "embedding_dim": d,
        }

    Rbar = R_norm / n

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

        if (
            not np.isfinite(kappa_new)
            or kappa_new <= 0
            or abs(step) > 0.5 * max(1.0, kappa)
        ):
            ok_newton = False
            break

        kappa = kappa_new

    if not ok_newton:
        lo = 0.0
        hi = _kappa_upper_bound(Rbar, d)

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            A_mid = _A_p_stable(mid, d)

            if A_mid < Rbar:
                lo = mid
            else:
                hi = mid

        kappa = 0.5 * (lo + hi)

    trace_var = float(max(1.0 - Rbar**2, 0.0))
    dcu = float(1.0 / kappa) if kappa > 0 else float("inf")

    return {
        "kappa": float(kappa),
        "dcu": dcu,
        "trace_var": trace_var,
        "mean_resultant_length": float(Rbar),
        "n_generations": n,
        "embedding_dim": d,
    }


# -----------------------------
# Sentence-BERT embedder
# -----------------------------
class SBERTEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings.")

        if len(texts) == 0:
            raise ValueError("No texts provided for embedding.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float64)


# -----------------------------
# DCU Calculator
# -----------------------------
class DCUCalculator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.embedder = SBERTEmbedder(
            model_name=model_name,
            device=device,
            cache_folder=cache_folder,
        )

        self.batch_size = batch_size

    def compute_from_generations(
        self,
        generations: List[str],
    ) -> Dict[str, Any]:

        if not isinstance(generations, list):
            raise TypeError("generations must be a list of strings.")

        cleaned = [
            normalize_text(g)
            for g in generations
            if isinstance(g, str) and g.strip()
        ]

        if len(cleaned) == 0:
            raise ValueError("No valid non-empty generations found.")

        embeddings = self.embedder.encode(
            cleaned,
            batch_size=self.batch_size,
        )

        stats = compute_vmf_stats(embeddings)

        return {
            "dcu": stats["dcu"],
            "kappa": stats["kappa"],
            "trace_var": stats["trace_var"],
            "mean_resultant_length": stats["mean_resultant_length"],
            "n_generations": stats["n_generations"],
            "embedding_dim": stats["embedding_dim"],
            "embedding_model": self.embedder.model_name,
        }

    def compute_from_json(
        self,
        data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:

        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict):
            if "generations" not in data:
                raise KeyError("JSON object must contain a 'generations' field.")

            result = self.compute_from_generations(data["generations"])

            return {
                **data,
                "dcu_result": result,
            }

        if isinstance(data, list):
            outputs = []

            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    raise TypeError(f"Item {idx} must be a dictionary.")

                if "generations" not in item:
                    raise KeyError(f"Item {idx} missing 'generations' field.")

                result = self.compute_from_generations(item["generations"])

                outputs.append({
                    **item,
                    "dcu_result": result,
                })

            return outputs

        raise TypeError("Input must be a JSON string, dictionary, or list of dictionaries.")


if __name__ == "__main__":
    print(
        "DCU package loaded successfully.\n"
        "Import DCUCalculator and call compute_from_json()."
    )