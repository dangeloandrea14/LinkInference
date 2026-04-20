"""
generate_configs.py  —  Generate minimal Gold Model screening configs.

Creates Identity + Gold Model configs at 1%, 5%, 20% forget-set sizes
for each candidate dataset (GCN only).
Output: configs/dataset_selection/<Dataset>/<Dataset>_GCN_<pct>.jsonc
"""

import json, os

DATASETS = {
    "RomanEmpire": {
        "datasource_class": "torch_geometric.datasets.HeterophilousGraphDataset",
        "datasource_params": {"root": "resources/data", "name": "roman-empire"},
        "in_channels": 300,
        "out_channels": 18,
    },
    "AmazonRatings": {
        "datasource_class": "torch_geometric.datasets.HeterophilousGraphDataset",
        "datasource_params": {"root": "resources/data", "name": "amazon-ratings"},
        "in_channels": 300,
        "out_channels": 5,
    },
    "Minesweeper": {
        "datasource_class": "torch_geometric.datasets.HeterophilousGraphDataset",
        "datasource_params": {"root": "resources/data", "name": "minesweeper"},
        "in_channels": 7,
        "out_channels": 2,
    },
    "Flickr": {
        "datasource_class": "torch_geometric.datasets.Flickr",
        "datasource_params": {"root": "resources/data/Flickr"},
        "in_channels": 500,
        "out_channels": 7,
    },
    "Penn94": {
        "datasource_class": "torch_geometric.datasets.LINKXDataset",
        "datasource_params": {"root": "resources/data", "name": "penn94"},
        "in_channels": 5,
        "out_channels": 2,
    },
    "ogbn-arxiv": {
        "datasource_class": "ogb.nodeproppred.PygNodePropPredDataset",
        "datasource_params": {"root": "resources/data", "name": "ogbn-arxiv"},
        "in_channels": 128,
        "out_channels": 40,
    },
    # --- New candidates ---
    "arxiv-year": {
        "datasource_class": "torch_geometric.datasets.LINKXDataset",
        "datasource_params": {"root": "resources/data", "name": "arxiv-year"},
        "in_channels": 128,
        "out_channels": 5,
    },
    "twitch-gamers": {
        "datasource_class": "torch_geometric.datasets.LINKXDataset",
        "datasource_params": {"root": "resources/data", "name": "twitch-gamers"},
        "in_channels": 7,
        "out_channels": 2,
    },
    "tolokers": {
        "datasource_class": "torch_geometric.datasets.HeterophilousGraphDataset",
        "datasource_params": {"root": "resources/data", "name": "tolokers"},
        "in_channels": 10,
        "out_channels": 2,
    },
    "DBLP": {
        "datasource_class": "torch_geometric.datasets.CitationFull",
        "datasource_params": {"root": "resources/data", "name": "DBLP"},
        "in_channels": 1639,
        "out_channels": 4,
    },
    # --- Existing benchmark datasets (not yet screened) ---
    "AmazonComputers": {
        "datasource_class": "torch_geometric.datasets.Amazon",
        "datasource_params": {"root": "resources/data", "name": "Computers"},
        "in_channels": 767,
        "out_channels": 10,
    },
    "AmazonPhotos": {
        "datasource_class": "torch_geometric.datasets.Amazon",
        "datasource_params": {"root": "resources/data", "name": "Photo"},
        "in_channels": 745,
        "out_channels": 8,
    },
    "ogbn-products": {
        "datasource_class": "ogb.nodeproppred.PygNodePropPredDataset",
        "datasource_params": {"root": "resources/data", "name": "ogbn-products"},
        "in_channels": 100,
        "out_channels": 47,
        "batched": True,
    },
}

FORGET_PCTS = {1: 0.01, 5: 0.05, 20: 0.20}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "output/runs/dataset_selection"


def make_config(ds_name, ds_cfg, pct_label, pct_frac):
    safe_name = ds_name.replace("-", "_")
    out_file = f"{OUTPUT_DIR}/{safe_name}_GCN_{pct_label}.json"

    datasource_params = dict(ds_cfg["datasource_params"])
    predictor_class = (
        "erasure.model.TorchGraphModelBatched.TorchGraphModelBatched"
        if ds_cfg.get("batched")
        else "erasure.model.TorchGraphModel.TorchGraphModel"
    )

    return f"""\
{{
    "data": {{"class":"erasure.data.datasets.DatasetManager.DatasetManager",
    "parameters": {{
        "DataSource": {{"class":"erasure.data.data_sources.TorchGeometricDataSource.TorchGeometricDataSource",
                        "parameters": {{"datasource": {{"class": "{ds_cfg['datasource_class']}", "parameters":{json.dumps(datasource_params)},
                        "preprocess":[]
                    }}
                    }}}},
        "partitions":  [
          {{"class":"erasure.data.datasets.DataSplitterGraph.DataSplitterPercentage", "parameters":{{"parts_names":["all_shuffled","-"], "percentage":1, "shuffle":true}}}},
          {{"class":"erasure.data.datasets.DataSplitterGraph.DataSplitterPercentage", "parameters":{{"parts_names":["train_0","test"], "percentage":0.8, "ref_data":"all_shuffled", "shuffle":false}}}},
          {{"class":"erasure.data.datasets.DataSplitterGraph.DataSplitterPercentage", "parameters":{{"parts_names":["validation","train"], "percentage":0.1, "ref_data":"train_0", "shuffle":false}}}},
          {{"class":"erasure.data.datasets.DataSplitterGraph.DataSplitterPercentage", "parameters":{{"parts_names":["forget","retain"], "percentage":{pct_frac}, "ref_data":"all", "shuffle":false, "edge_removal":true}}}}
                    ],
        "batch_size":4,
        "split_seed":16}}
    }},
    "predictor": {{
        "class": "{predictor_class}",
        "parameters": {{
          "epochs": 100,
          "optimizer": {{
            "class": "torch.optim.Adam",
            "parameters": {{"lr": 0.001}}
          }},
          "loss_fn": {{
            "class": "torch.nn.CrossEntropyLoss",
            "parameters": {{"reduction":"mean"}}
          }},
          "model": {{
            "class": "erasure.model.graphs.GCN.GCN",
            "parameters": {{"in_channels":{ds_cfg['in_channels']}, "hidden_channels":[64], "out_channels":{ds_cfg['out_channels']}}}
          }}
        }}
      }},

    "unlearners":[
        // Original Model
              {{"compose_idt" : "configs/snippets/u_id.json"}},
        // Gold Model
              {{
        "class": "erasure.unlearners.GoldModel.GoldModelGraph",
        "parameters": {{
          "training_set": "retain"
        }}
        }}
    ],

      "evaluator":{{
          "class": "erasure.evaluations.manager.Evaluator",
          "parameters": {{
            "measures":[
              {{"class":"erasure.evaluations.running.RunTime"}},
              {{"class": "erasure.evaluations.measures.TorchSKLearnGraph", "parameters":{{"partition":"test","target":"unlearned"}}}},
              {{"class": "erasure.evaluations.measures.TorchSKLearnGraph", "parameters":{{"partition":"test","target":"unlearned", "unlearned_graph":false}}}},
              {{"class": "erasure.evaluations.measures.TorchSKLearnGraph", "parameters":{{"partition":"test","target":"original", "unlearned_graph":false}}}},
              {{"class": "erasure.evaluations.measures.SaveValues", "parameters":{{"path":"{out_file}"}}}}
            ]
          }}
        }},
    "globals":{{"cached": "false","seed": 0, "removal_type":"edge"}}
  }}"""


for ds_name, ds_cfg in DATASETS.items():
    folder = os.path.join(BASE_DIR, ds_name)
    os.makedirs(folder, exist_ok=True)
    for pct_label, pct_frac in FORGET_PCTS.items():
        safe_name = ds_name.replace("-", "_")
        fname = f"{safe_name}_GCN_{pct_label}.jsonc"
        path = os.path.join(folder, fname)
        content = make_config(ds_name, ds_cfg, pct_label, pct_frac)
        with open(path, "w") as f:
            f.write(content + "\n")
        print(f"  wrote {path}")

print("Done.")
