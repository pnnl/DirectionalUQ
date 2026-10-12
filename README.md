# directional-uq

## Overview

This repository contains the reference implementation and experiments for **Directional Concentration Uncertainty (DCU)**, a representational approach to uncertainty quantification for generative models. DCU treats a set of model generations as points on a hypersphere, fits a von Mises–Fisher (vMF) distribution to their sentence/text embeddings, and uses the estimated concentration parameter (kappa) and its derived variance quantities as an uncertainty signal. The scripts and notebooks below implement the DCU estimator and evaluate it across several question-answering and multimodal captioning benchmarks, comparing it against semantic-entropy baselines.

Paper: [Directional Concentration Uncertainty: A representational approach to uncertainty quantification for generative models (arXiv:2602.13264)](https://arxiv.org/abs/2602.13264)

Citation:

> Chattopadhyay, S., Kennedy, B., Munikoti, S., Sarkar, S., & Pazdernik, K. (2026). Directional Concentration Uncertainty: A representational approach to uncertainty quantification for generative models. arXiv preprint arXiv:2602.13264.

### Scripts and notebooks

- **`DirectionalConsistencyMeasure.py`** — Standalone, packaged implementation of the DCU metric. Provides numerically stable vMF helpers (`_A_p_stable`, Newton iteration with bisection fallback) to estimate the concentration `kappa` and the DCU score (`1/kappa`) from a set of generations. Exposes an `SBERTEmbedder` (Sentence-BERT) and a `DCUCalculator` class with `compute_from_generations` / `compute_from_json` entry points for scoring generations directly or from JSON records.
- **`uq_direction_semantic.py`** — Dataset-agnostic experiment runner shared by the QA notebooks. Loads a causal LLM, samples multiple completions per prompt, embeds them (e.g., `intfloat/e5-large-v2`), fits vMF statistics (`kappa`, `trace_var`, `var_parallel`, `var_perp`), and optionally computes a DeBERTa-based semantic-entropy baseline. The `run_uq` function drives generation + scoring over an arbitrary `DataFrame` given a prompt template/function and writes results to CSV.
- **`concentration_squad (1).ipynb`** — Exploratory notebook on the SQuAD dataset that generates completions with Llama-2-7b-chat, computes vMF concentration statistics, and experiments with semantic clustering (spectral clustering, cosine similarity, adjusted Rand index) to relate directional concentration to answer consistency.
- **`squad_all_models_300.ipynb`** — Runs the DCU pipeline (`uq_direction_semantic`) on ~300 SQuAD questions across multiple LLMs using context-aware prompts, then evaluates uncertainty against ROUGE-L correctness via ROC-AUC.
- **`triviaqa_all_models_300.ipynb`** — Applies the same DCU pipeline to ~300 TriviaQA questions across multiple LLMs (Llama-2, Falcon, Mistral) and scores the resulting uncertainty measures against ROUGE-L-based correctness with AUROC and bootstrap analysis.
- **`nqopen_all_models_300.ipynb`** — Runs the DCU pipeline on ~300 Natural Questions (NQ-Open) examples across multiple LLMs and evaluates the uncertainty signals against ROUGE-L correctness.
- **`sciqa_allmodels_150.ipynb`** — Multimodal experiment on ScienceQA (~150 items) using vision-language backends (LLaVA / Gemma3 / Phi-4). Generates multiple answers per image+question, embeds them with a CLIP text embedder, computes vMF/DCU statistics, and compares against semantic entropy.
- **`arxivcap_magma.ipynb`** — Multimodal captioning experiment on the ArxivCap figure-caption dataset. Builds figure/caption pairs, generates captions with vision-language models, embeds generations with CLIP, and computes DCU alongside a DeBERTa semantic-entropy baseline.

# License

Copyright Battelle Memorial Institute 2026

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Disclaimer:

This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

                PACIFIC NORTHWEST NATIONAL LABORATORY
                             operated by
                               BATTELLE
                               for the
                  UNITED STATES DEPARTMENT OF ENERGY
                   under Contract DE-AC05-76RL01830


