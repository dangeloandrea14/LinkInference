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
        self.top_k  = self.local_config["parameters"].get("top_k", None)

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
        src_raw    = edges_df["numeric_id_1"].values
        dst_raw    = edges_df["numeric_id_2"].values

        # ── top-k language filtering (language target only) ───────────────────
        if self.target == "language" and self.top_k is not None:
            counts      = np.bincount(lang_codes.astype(np.int64))
            top_codes   = np.argsort(counts)[::-1][:self.top_k]
            keep_mask   = np.isin(lang_codes.astype(np.int64), top_codes)

            # remap old node indices → new consecutive indices (-1 = removed)
            old2new = np.full(len(df), -1, dtype=np.int64)
            old2new[keep_mask] = np.arange(keep_mask.sum())

            # remap labels to 0..top_k-1 in descending-frequency order
            code_remap = np.full(int(lang_codes.max()) + 1, -1, dtype=np.int64)
            for new_code, old_code in enumerate(top_codes):
                code_remap[old_code] = new_code
            y = torch.tensor(code_remap[lang_codes.astype(np.int64)[keep_mask]], dtype=torch.long)

            x = x[keep_mask]

            # filter edges: keep only those where both endpoints survive
            edge_keep = (old2new[src_raw] >= 0) & (old2new[dst_raw] >= 0)
            src_raw   = old2new[src_raw[edge_keep]]
            dst_raw   = old2new[dst_raw[edge_keep]]

        src = torch.tensor(src_raw, dtype=torch.long)
        dst = torch.tensor(dst_raw, dtype=torch.long)
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
