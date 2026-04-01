import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from erasure.data.data_sources.datasource import DataSource
from erasure.data.data_sources.EdgeFileDataSource import GeometricWrapper
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local


_REFERENCE_DATE = pd.Timestamp("2009-01-01")


class SingleGraphDataset:
    """Wraps a single PyG Data object to behave like a PyG single-graph dataset.

    GeometricWrapper (and DataSplitterGraph) expect `wrapper.data` to be an
    object with `.x`, `.y`, `.edge_index` attributes, matching how PyG datasets
    (e.g. Planetoid) expose their underlying Data via attribute forwarding.
    """

    def __init__(self, data: Data):
        self._data = data

    # Attribute forwarding — mirrors PyG dataset behaviour
    @property
    def x(self):           return self._data.x
    @property
    def y(self):           return self._data.y
    @property
    def edge_index(self):  return self._data.edge_index
    @property
    def edge_attr(self):   return self._data.edge_attr
    @property
    def num_nodes(self):   return self._data.num_nodes

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self._data

    def __iter__(self):
        yield self._data


class TwitchGamersDataSource(DataSource):
    """
    DataSource for the large Twitch Gamers dataset.

    Expects two CSV files under `root`:
      - large_twitch_features.csv
      - large_twitch_edges.csv

    target="mature" (default): binary label (0/1, ~balanced 53/47).
      Features (7-dim): views, life_time, created_at, updated_at,
                        dead_account, language (encoded), affiliate.

    target="language": 21-class label (language code → integer).
      Features (6-dim): same as above but language dropped from features.
    """

    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.root   = self.local_config["parameters"]["root"]
        self.name   = self.local_config["parameters"].get("name", "TwitchGamers")
        self.target = self.local_config["parameters"].get("target", "mature")

    def get_name(self):
        return self.name

    def create_data(self):
        feat_path  = os.path.join(self.root, "large_twitch_features.csv")
        edges_path = os.path.join(self.root, "large_twitch_edges.csv")

        # ── node features & labels ────────────────────────────────────────────
        df = pd.read_csv(feat_path)

        lang_codes = pd.Categorical(df["language"]).codes.astype(np.float32)

        if self.target == "language":
            y = torch.tensor(lang_codes.astype(np.int64), dtype=torch.long)
        else:
            y = torch.tensor(df["mature"].values, dtype=torch.long)

        # continuous
        views     = np.log1p(df["views"].values).astype(np.float32)
        life_time = np.log1p(df["life_time"].values).astype(np.float32)
        created   = (pd.to_datetime(df["created_at"]) - _REFERENCE_DATE).dt.days.values.astype(np.float32) / 3650.0
        updated   = (pd.to_datetime(df["updated_at"]) - _REFERENCE_DATE).dt.days.values.astype(np.float32) / 3650.0

        # binary
        dead      = df["dead_account"].values.astype(np.float32)
        affiliate = df["affiliate"].values.astype(np.float32)

        if self.target == "language":
            # drop language from features — model must infer it from graph structure
            feature_cols = [views, life_time, created, updated, dead, affiliate]
        else:
            feature_cols = [views, life_time, created, updated, dead, lang_codes, affiliate]

        x = torch.tensor(np.stack(feature_cols, axis=1), dtype=torch.float32)

        # ── edges ─────────────────────────────────────────────────────────────
        edges_df   = pd.read_csv(edges_path)
        src        = torch.tensor(edges_df["numeric_id_1"].values, dtype=torch.long)
        dst        = torch.tensor(edges_df["numeric_id_2"].values, dtype=torch.long)
        # store both directions (undirected graph)
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )

        data = Data(x=x, edge_index=edge_index, y=y)
        return GeometricWrapper(SingleGraphDataset(data), self.preprocess)

    def get_simple_wrapper(self, data):
        return GeometricWrapper(data, self.preprocess)

    def check_configuration(self):
        super().check_configuration()
