---
title: "Edge Removal Benchmark — Configuration Report"
date: "March 2026"
geometry: margin=2.5cm
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{xcolor}
  - \usepackage{textcomp}
  - \definecolor{warncolor}{RGB}{180,0,0}
  - \newcommand{\warn}[1]{\textcolor{warncolor}{\textbf{#1}}}
---

# Edge Removal Benchmark — Configuration Report

## 1. Datasets

Eight datasets are tested, all benchmarked across the full set of architectures.

| Dataset | Source | Features | Classes | Split seed |
|---|---|---:|---:|---:|
| AmazonComputers | `tg.Amazon` (Computers) | 767 | 10 | 16 |
| AmazonPhotos | `tg.Amazon` (Photos) | 767 | 10 | 16 |
| Citeseer | `tg.Planetoid` (Citeseer) | 3703 | 6 | 16 |
| Cora | `tg.Planetoid` (Cora) | 1433 | 7 | 16 |
| Pubmed | `tg.Planetoid` (Pubmed) | 500 | 3 | 16 |
| ogbn-arxiv | `ogb.PygNodePropPredDataset` | 128 | 40 | 16 |
| ogbn-products | `ogb.PygNodePropPredDataset` | 100 | 47 | 16 |
| Synthetic | `tg.FakeDataset` | 64 | 2 | 16 |

`tg` = `torch_geometric.datasets`, `ogb` = `ogb.nodeproppred`

**Synthetic dataset parameters:** `num_graphs=1`, `avg_num_nodes=1000`, `avg_degree=50`, `num_classes=2`, `task="node"`, preprocessed with `MakeCentroidFeatures`.

---

## 2. Forget Set Extraction

All configurations use the same four-stage data splitting pipeline. The final stage produces the forget/retain sets via `edge_removal=True`.

| Stage | Output splits | % | Source | Shuffle |
|---|---|---|---|---|
| 1 | `all_shuffled` / `-` | 100% | full dataset | **true** |
| 2 | `train_0` / `test` | 80% / 20% | `all_shuffled` | false |
| 3 | `validation` / `train` | 10% / 90% | `train_0` | false |
| 4 | `forget` / `retain` | **X%** / rest | `train` | false |

Stage 4 sets `edge_removal=True`, meaning sampled indices represent **edges to be forgotten** rather than nodes.

**Forget percentages tested per dataset:**

| Dataset | Forget % variants |
|---|---|
| AmazonComputers | 5%, 20%, 50%, 99%, 100% (GCN); 5%, 20% (all other archs) |
| AmazonPhotos | 5%, 20%, 50%, 99%, 100% (GCN); 5%, 20% (all other archs) |
| Citeseer | 5%, 20%, 50%, 99%, 100% (GCN); 5%, 20% (all other archs) |
| Cora | 5%, 20%, 50%, 99%, 100% (GCN); 5%, 20% (all other archs) |
| Pubmed | 5%, 20%, 50%, 99%, 100% (GCN); 5%, 20% (all other archs) |
| ogbn-arxiv | 5%, 20%, 50%, 99%, 100% (GCN); 5%, 20% (all other archs) |
| ogbn-products | 5%, 20% (all archs) |
| Synthetic | 5%, 20% (all archs) |

The five GCN variants differ **only** in forget percentage; all other parameters are identical. All non-GCN architectures now also have a 5% variant alongside the existing 20% variant.

---

## 3. Model Architectures

All models use hidden size 64 (one hidden layer), Adam optimizer (lr=0.001), CrossEntropyLoss, and a linear LR decay scheduler.

**Standard datasets** (all 6 main datasets) use `TorchGraphModel`:

| Architecture | Config suffix | Notes |
|---|---|---|
| GAT | `_GAT` | Graph Attention Network |
| GCN | `_GCN_20/50/99/100` | Graph Convolutional Network; suffix = forget % |
| GIN | `_GIN` | Graph Isomorphism Network |
| GraphSAGE | `_GraphSAGE` | |
| MLP | `_MLP` | No graph structure used |
| SGC | `_SGC` | Simple Graph Convolution |
| SGC\_CGU | `_SGC_CGU` | SGC variant for CGU compatibility |

**Training:** 100 epochs, early stopping (patience=10, min\_delta=0.01), best weights restored.

**ogbn-products** uses `TorchGraphModelBatched` with mini-batch training via `NeighborLoader` (fanouts=[15,10], batch\_size=1024). Early stopping is threshold-based (not patience-based).

**Synthetic** uses `TorchGraphModel` with patience-based early stopping (same as the 6 main datasets).

---

## 4. Unlearners

All 6 main datasets and Pubmed share the same set of **17 unlearners**. ogbn-products and Synthetic expand each gradient-based unlearner to **3 learning rate variants** (lr in {0.001, 0.0001, 0.00001}), plus 3 Fisher alpha variants, yielding ~54 entries total.

| # | Unlearner | Key parameters |
|---|---|---|
| 1 | **Original** | Identity (no unlearning) |
| 2 | **Gold Model** | Retrain from scratch on retain set |
| 3 | **Fine-Tuning (FT)** | 1 epoch on retain, lr=0.001 |
| 4 | **Successive Random Labels (SRL)** | 1 epoch, lr=0.001 |
| 5 | **cfk** | FT, last 2 layers, 1 epoch, lr=0.001 |
| 6 | **eu\_k** | Last 2 layers, 10 epochs on retain, lr=0.001 |
| 7 | **NegGrad** | 1 epoch gradient ascent on forget, lr=0.001 |
| 8 | **AdvancedNegGrad** | 1 epoch, retain+forget, lr=0.001 |
| 9 | **UNSIR → FT** (Cascade) | noise\_lr=0.1, 1 epoch each, lr=0.001 |
| 10 | **BadTeaching** | 1 epoch, KL\_temperature=1.0, lr=0.001 |
| 11 | **SCRUB** | 1 epoch, T=2.0, lr=0.001 |
| 12 | **Fisher Forgetting** | alpha=1e-6 |
| 13 | **Selective Synaptic Dampening (SSD)** | dampening\_constant=0.1, selection\_weighting=50, lr=0.001 |
| 14 | **SalUn** (SaliencyMap → SRL Cascade) | threshold=0.5, 1 epoch, lr=0.001 |
| 15 | **IDEA** | scale=5e4 |
| 16 | **CGU\_edge** | Certified graph unlearning (edge) |
| 17 | **CEU** | Certified edge unlearning |

---

## 5. Evaluation Metrics

All datasets share the same evaluation suite:

| Metric | Partitions / targets |
|---|---|
| Runtime | — |
| Accuracy | test/forget/train × unlearned; test × original |
| F1-macro | test/forget × unlearned; test/forget × original |
| AUS (Anamnesis Unlearning Score) | forget vs. test |
| UMIA | (composed snippet) |
| LinkTeller | unlearned, original |
| LinkStealing (Attack 0) | unlearned |

All configurations share: `seed=0`, `removal_type="edge"`, `cached=false`.

---

## 6. Summary of Setup Differences

All datasets now share a uniform setup. The only remaining differences are:

| Property | Main 6 datasets + Pubmed + Synthetic | ogbn-products |
|---|---|---|
| Architectures tested | 10 (GAT, GCN×5, GIN, SAGE, MLP, SGC, SGC\_CGU) | 10 (same) |
| Training class | `TorchGraphModel` | `TorchGraphModelBatched` |
| Epochs | 100 | 100 |
| Early stopping | Patience-based (p=10, min\_delta=0.01) | Threshold-based |
| Unlearners | 17 (standard) | 17 (standard) |
| Evaluation suite | Standard | Standard |
| Forget % variants | 5/20/50/99/100 (GCN); 5/20% (others) | 5/20/50/99/100 (GCN); 5/20% (others) |
| Hidden channels | 64 | 64 |

`TorchGraphModelBatched` is required for ogbn-products due to its graph size (mini-batch training via `NeighborLoader`, fanouts=[15,10], batch\_size=1024).
