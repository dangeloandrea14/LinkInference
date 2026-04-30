#!/usr/bin/env python3
"""Generate EdgeUnbench benchmark configs for 5 datasets × 7 architectures × hard/easy splits."""

import os

BASE_DIR = "configs/benchmark/EdgeUnbench"

DATASETS = {
    "AmazonPhotos": {
        "datasource_class": "torch_geometric.datasets.Amazon",
        "datasource_params": '{"root":"resources/data", "name":"Photo"}',
        "in_channels": 745,
        "out_channels": 8,
        "predictor_class": "erasure.model.TorchGraphModel.TorchGraphModel",
        "batch_size": 4,
        "batched": False,
    },
    "Flickr": {
        "datasource_class": "torch_geometric.datasets.Flickr",
        "datasource_params": '{"root":"resources/data/Flickr"}',
        "in_channels": 500,
        "out_channels": 7,
        "predictor_class": "erasure.model.TorchGraphModel.TorchGraphModel",
        "batch_size": 4,
        "batched": False,
    },
    "Reddit": {
        "datasource_class": "torch_geometric.datasets.Reddit",
        "datasource_params": '{"root":"resources/data/Reddit"}',
        "in_channels": 602,
        "out_channels": 41,
        "predictor_class": "erasure.model.TorchGraphModelBatched.TorchGraphModelBatched",
        "batch_size": 1024,
        "batched": True,
    },
    "RomanEmpire": {
        "datasource_class": "torch_geometric.datasets.HeterophilousGraphDataset",
        "datasource_params": '{"root":"resources/data", "name":"roman-empire"}',
        "in_channels": 300,
        "out_channels": 18,
        "predictor_class": "erasure.model.TorchGraphModel.TorchGraphModel",
        "batch_size": 4,
        "batched": False,
    },
    "ogbn-arxiv": {
        "datasource_class": "ogb.nodeproppred.PygNodePropPredDataset",
        "datasource_params": '{"root":"resources/data", "name":"ogbn-arxiv"}',
        "in_channels": 128,
        "out_channels": 40,
        "predictor_class": "erasure.model.TorchGraphModel.TorchGraphModel",
        "batch_size": 4,
        "batched": False,
    },
}

ARCHITECTURES = {
    "GCN":       "erasure.model.graphs.GCN.GCN",
    "GIN":       "erasure.model.graphs.GIN.GIN",
    "GAT":       "erasure.model.graphs.GAT.GAT",
    "GraphSAGE": "erasure.model.graphs.GraphSAGE.GraphSAGE",
    "SGC":       "erasure.model.graphs.SGC.SGC",
    "MLP":       "erasure.model.graphs.MLP.MLP",
    "SGC_CGU":   "erasure.model.graphs.SGC_CGU.SGC_CGU",
}

UNLEARNERS = """    "unlearners":[
        // Original Model
        {"compose_idt" : "configs/snippets/u_id.json"},
        // Gold Model
        {
            "class": "erasure.unlearners.GoldModel.GoldModelGraph",
            "parameters": {"training_set": "retain"}
        },
        // Fine-Tuning
        {
            "class": "erasure.unlearners.graph_unlearners.Finetuning.Finetuning",
            "parameters": {
                "epochs": 1,
                "ref_data":"retain",
                "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
            }
        },
        // Successive Random Labelling
        {
            "class": "erasure.unlearners.graph_unlearners.SuccessiveRandomLabels.SuccessiveRandomLabels",
            "parameters": {
                "epochs": 1,
                "ref_data_retain": "retain",
                "ref_data_forget": "forget",
                "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
            }
        },
        // cfk
        {
            "class": "erasure.unlearners.graph_unlearners.Finetuning.Finetuning",
            "parameters": {
                "last_trainable_layers": 2,
                "epochs": 1,
                "ref_data":"retain",
                "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
            }
        },
        // euk
        {
            "class": "erasure.unlearners.graph_unlearners.eu_k.eu_k",
            "parameters": {
                "last_trainable_layers": 2,
                "epochs": 10,
                "ref_data":"retain",
                "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
            }
        },
        // neggrad
        {
            "class": "erasure.unlearners.graph_unlearners.NegGrad.NegGrad",
            "parameters": {
                "epochs": 1,
                "ref_data": "forget",
                "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
            }
        },
        // advanced neggrad
        {
            "class": "erasure.unlearners.graph_unlearners.AdvancedNegGrad.AdvancedNegGrad",
            "parameters": {
                "epochs": 1,
                "ref_data_retain": "retain",
                "ref_data_forget": "forget",
                "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
            }
        },
        // UNSIR
        {
            "class": "erasure.unlearners.composite.Cascade",
            "parameters": {
                "sub_unlearner": [
                    {
                        "class": "erasure.unlearners.graph_unlearners.UNSIR.UNSIR",
                        "parameters": {
                            "epochs": 1,
                            "ref_data_retain": "retain",
                            "ref_data_forget": "forget",
                            "noise_lr": 0.1,
                            "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
                        }
                    },
                    {
                        "class": "erasure.unlearners.graph_unlearners.Finetuning.Finetuning",
                        "parameters": {
                            "epochs": 1,
                            "ref_data":"retain",
                            "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
                        }
                    }
                ]
            }
        },
        // Bad Teaching
        {
            "class": "erasure.unlearners.graph_unlearners.BadTeaching.BadTeaching",
            "parameters": {
                "epochs": 1,
                "ref_data_retain": "retain",
                "ref_data_forget": "forget",
                "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}},
                "KL_temperature": 1.0
            }
        },
        // SCRUB
        {
            "class": "erasure.unlearners.graph_unlearners.Scrub.Scrub",
            "parameters": {
                "epochs": 1,
                "ref_data_retain": "retain",
                "ref_data_forget": "forget",
                "T": 2.0,
                "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
            }
        },
        // Fisher Forgetting
        {
            "class": "erasure.unlearners.graph_unlearners.FisherForgetting.FisherForgetting",
            "parameters": {
                "ref_data":"retain",
                "alpha": 1e-6
            }
        },
        // Selective Synaptic Dampening
        {
            "class": "erasure.unlearners.graph_unlearners.SelectiveSynapticDampening.SelectiveSynapticDampening",
            "parameters": {
                "ref_data_train": "train",
                "ref_data_forget": "forget",
                "dampening_constant": 0.1,
                "selection_weighting": 50,
                "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
            }
        },
        // SalUn
        {
            "class": "erasure.unlearners.composite.Cascade",
            "parameters": {
                "sub_unlearner": [
                    {
                        "class": "erasure.unlearners.graph_unlearners.SaliencyMapGeneration.SaliencyMapGeneration",
                        "parameters": {
                            "ref_data":"forget",
                            "treshold": 0.5,
                            "save_dir": "saliency_maps",
                            "file_name": "saliency_map"
                        }
                    },
                    {
                        "class": "erasure.unlearners.graph_unlearners.SuccessiveRandomLabels.SuccessiveRandomLabels",
                        "parameters": {
                            "model_mask_path": "saliency_maps/saliency_map",
                            "epochs": 1,
                            "ref_data_retain": "retain",
                            "ref_data_forget": "forget",
                            "optimizer": {"class": "torch.optim.Adam","parameters": {"lr": 0.001}}
                        }
                    }
                ]
            }
        },
        {"class": "erasure.unlearners.certified_graph_unlearners.IDEA.IDEA", "parameters":{"scale":5e4}},
        {"class": "erasure.unlearners.certified_graph_unlearners.CGU.CGU_edge", "parameters": {}},
        {"class": "erasure.unlearners.certified_graph_unlearners.CEU.CEU", "parameters":{}},
        {
            "class": "erasure.unlearners.certified_graph_unlearners.ScaleGUN.ScaleGUN",
            "parameters": {
                "std": 1e-2,
                "lam": 1e-2,
                "eps": 1.0,
                "delta": 1e-4,
                "prop_step": 3,
                "num_steps_optimizer": 100,
                "lr": 0.01,
                "ref_data_retain": "retain",
                "ref_data_forget": "forget",
                "train_mode": "ovr"
            }
        },
        {
            "class": "erasure.unlearners.graph_unlearners.GNNDelete.GNNDelete",
            "parameters": {
                "epochs": 10,
                "lr": 1e-3,
                "alpha": 0.5,
                "loss_fct": "mse_mean",
                "ref_data_forget": "forget"
            }
        }
    ],"""

EVALUATOR_TEMPLATE = """    "evaluator":{
        "class": "erasure.evaluations.manager.Evaluator",
        "parameters": {
            "measures":[
                {"class":"erasure.evaluations.running.RunTime"},
                {"class": "erasure.evaluations.measures.TorchSKLearnGraph", "parameters":{"partition":"test","target":"unlearned"}},
                {"class": "erasure.evaluations.measures.TorchSKLearnGraph", "parameters":{"partition":"forget","target":"unlearned"}},
                {"class": "erasure.evaluations.measures.TorchSKLearnGraph", "parameters":{"partition":"train","target":"unlearned"}},
                {"class": "erasure.evaluations.measures.TorchSKLearnGraph", "parameters":{"partition":"test","target":"unlearned", "unlearned_graph":false}},
                {"class": "erasure.evaluations.measures.TorchSKLearnGraph", "parameters":{"partition":"test","target":"original", "unlearned_graph":false}},
                //F1 score
                {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
                "parameters": {"partition": "test", "target": "unlearned", "name": "f1_macro", "function": { "class": "sklearn.metrics.f1_score", "parameters": { "average": "macro" } }}},
                {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
                "parameters": {"partition": "test", "target": "original", "name": "f1_macro", "function": { "class": "sklearn.metrics.f1_score", "parameters": { "average": "macro" } }}},
                {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
                "parameters": {"partition": "forget", "target": "unlearned", "name": "f1_macro", "function": { "class": "sklearn.metrics.f1_score", "parameters": { "average": "macro" } }}},
                {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
                "parameters": {"partition": "forget", "target": "original", "name": "f1_macro", "function": { "class": "sklearn.metrics.f1_score", "parameters": { "average": "macro" } }}},
                {"class": "erasure.evaluations.measures.AUSGraph", "parameters":{"forget_part": "forget","test_part": "test"}},
                //UMIA
                {"compose_umia" : "configs/snippets/e_umia_graph.json"},
                //LinkTeller
                {"class": "erasure.evaluations.LinkTeller.LinkTeller.LinkTeller", "parameters":{"target":"unlearn"}},
                {"class": "erasure.evaluations.LinkTeller.LinkTeller.LinkTeller", "parameters":{"target":"original"}},
                //LinkStealing
                {"class":"erasure.evaluations.link_stealing_attack.link_stealing_attack_0.LinkStealing0", "parameters":{"target":"unlearned"}},
                {"class": "erasure.evaluations.measures.SaveValues", "parameters":{"path":"OUTPUT_PATH"}}
            ]
        }
    },"""


def make_predictor(ds_cfg, arch_class, in_channels, out_channels):
    if ds_cfg["batched"]:
        return f"""    "predictor": {{
        "class": "{ds_cfg['predictor_class']}",
        "parameters": {{
            "epochs": 100,
            "optimizer": {{"class": "torch.optim.Adam","parameters": {{"lr": 0.001}}}},
            "loss_fn": {{"class": "torch.nn.CrossEntropyLoss","parameters": {{"reduction":"mean"}}}},
            "model": {{
                "class": "{arch_class}",
                "parameters": {{"in_channels":{in_channels}, "hidden_channels":[64], "out_channels":{out_channels}}}
            }},
            "batch_size": {ds_cfg['batch_size']},
            "num_neighbors": [15, 10],
            "early_stopping_threshold": null
        }}
    }},"""
    else:
        return f"""    "predictor": {{
        "class": "{ds_cfg['predictor_class']}",
        "parameters": {{
            "epochs": 100,
            "optimizer": {{"class": "torch.optim.Adam","parameters": {{"lr": 0.001}}}},
            "loss_fn": {{"class": "torch.nn.CrossEntropyLoss","parameters": {{"reduction":"mean"}}}},
            "model": {{
                "class": "{arch_class}",
                "parameters": {{"in_channels":{in_channels}, "hidden_channels":[64], "out_channels":{out_channels}}}
            }}
        }}
    }},"""


def make_config(dataset_name, ds_cfg, arch_name, arch_class, difficulty):
    """Generate a full config string for dataset × architecture × difficulty."""
    output_path = f"output/runs/LinkAttack/edge/EdgeUnbench/{dataset_name}_{arch_name}_{difficulty}.json"

    data_section = f"""    "data": {{"class":"erasure.data.datasets.DatasetManager.DatasetManager",
    "parameters": {{
        "DataSource": {{"class":"erasure.data.data_sources.TorchGeometricDataSource.TorchGeometricDataSource",
                        "parameters": {{"datasource": {{"class": "{ds_cfg['datasource_class']}",
                        "parameters":{ds_cfg['datasource_params']},
                        "preprocess":[]
                    }}}}}},
        "partitions": [
            {{"class":"erasure.data.datasets.DataSplitterGraph.DataSplitterPercentage", "parameters":{{"parts_names":["all_shuffled","-"], "percentage":1, "shuffle":true}}}},
            {{"class":"erasure.data.datasets.DataSplitterGraph.DataSplitterPercentage", "parameters":{{"parts_names":["train_0","test"], "percentage":0.8, "ref_data":"all_shuffled", "shuffle":false}}}},
            {{"class":"erasure.data.datasets.DataSplitterGraph.DataSplitterPercentage", "parameters":{{"parts_names":["validation","train"], "percentage":0.1, "ref_data":"train_0", "shuffle":false}}}},
            {{"class":"erasure.data.datasets.DataSplitterGraph.DataSplitterEdgeDifficulty", "parameters":{{"parts_names":["forget","retain"], "percentage":0.05, "ref_data":"all", "mode":"{difficulty}"}}}}
        ],
        "batch_size":{ds_cfg['batch_size']},
        "split_seed":16}}
    }},"""

    predictor_section = make_predictor(
        ds_cfg, arch_class, ds_cfg["in_channels"], ds_cfg["out_channels"]
    )

    evaluator_section = EVALUATOR_TEMPLATE.replace("OUTPUT_PATH", output_path)

    config = "{\n"
    config += data_section + "\n"
    config += predictor_section + "\n\n"
    config += UNLEARNERS + "\n\n"
    config += evaluator_section + "\n"
    config += '    "globals":{"cached": "false","seed": 0, "removal_type":"edge"}\n'
    config += "}\n"
    return config


def main():
    created = 0
    for dataset_name, ds_cfg in DATASETS.items():
        folder = os.path.join(BASE_DIR, dataset_name)
        os.makedirs(folder, exist_ok=True)

        for arch_name, arch_class in ARCHITECTURES.items():
            # Hard config for every architecture
            cfg = make_config(dataset_name, ds_cfg, arch_name, arch_class, "hard")
            fname = os.path.join(folder, f"{dataset_name}_{arch_name}_hard.jsonc")
            with open(fname, "w") as f:
                f.write(cfg)
            print(f"  created {fname}")
            created += 1

            # Easy config only for GIN
            if arch_name == "GIN":
                cfg = make_config(dataset_name, ds_cfg, arch_name, arch_class, "easy")
                fname = os.path.join(folder, f"{dataset_name}_{arch_name}_easy.jsonc")
                with open(fname, "w") as f:
                    f.write(cfg)
                print(f"  created {fname}")
                created += 1

    print(f"\nTotal configs created: {created}")


if __name__ == "__main__":
    main()
