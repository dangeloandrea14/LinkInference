from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, negative_sampling


class DeletionLayer(nn.Module):
    """Learnable linear transformation applied only to infected (masked) nodes."""

    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.mask
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)
            return new_rep
        return x


class GNNDelete(GraphUnlearner):

    def init(self):
        """
        Initializes GNNDelete with global and local contexts.
        """
        super().init()

        self.epochs = self.local.config['parameters']['epochs']
        self.lr = self.local.config['parameters']['lr']
        self.alpha = self.local.config['parameters']['alpha']
        self.loss_fct_name = self.local.config['parameters']['loss_fct']
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget']

    def _get_loss_fct(self, name):
        if name == 'mse_mean':
            return nn.MSELoss(reduction='mean')
        elif name == 'mse_sum':
            return nn.MSELoss(reduction='sum')
        elif name == 'kld_mean':
            return lambda logits, truth: 1 - torch.exp(
                -F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'batchmean'))
        elif name == 'cosine_mean':
            return lambda logits, truth: (1 - F.cosine_similarity(logits, truth)).mean()
        else:
            raise NotImplementedError(f'Unknown loss function: {name}')

    def __unlearn__(self):
        """
        An implementation of the GNNDelete unlearning algorithm proposed in the following paper:
        "Cheng, Jiali, et al. 'GNNDelete: A General Strategy for Unlearning in Graph Neural Networks.'
        International Conference on Learning Representations, 2023."

        Codebase taken from the original implementation: https://github.com/mims-harvard/GNNDelete

        Deletion layers are injected into the existing model via forward hooks rather than subclassing,
        so GNNDelete works with any GNN architecture without requiring model code changes.
        """
        self.info('Starting GNNDelete')

        forget_edges = self.dataset.partitions[self.ref_data_forget]
        num_nodes = self.x.size(0)
        model = self.predictor.model

        # directed_df_edge_index: forget edges as [2, n_forget] (directed, as stored in the partition)
        directed_df_edge_index = torch.tensor(forget_edges, dtype=torch.long, device=self.device).t()

        # df_mask: boolean over self.edge_index, True for forget edges (both directions)
        forget_set = set(map(tuple, forget_edges)) | {(v, u) for u, v in forget_edges}
        edge_list = self.edge_index.t().cpu().tolist()
        df_mask = torch.tensor([tuple(e) in forget_set for e in edge_list],
                               dtype=torch.bool, device=self.device)
        dr_mask = ~df_mask

        # k-hop subgraph masks for S_Df (the subgraph defined by the forget edges)
        forget_nodes = directed_df_edge_index.flatten().unique()

        _, two_hop_edges, _, sdf_mask = k_hop_subgraph(
            forget_nodes, 2, self.edge_index, num_nodes=num_nodes)
        _, one_hop_edges, _, _ = k_hop_subgraph(
            forget_nodes, 1, self.edge_index, num_nodes=num_nodes)

        sdf_node_1hop_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        sdf_node_2hop_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        sdf_node_1hop_mask[one_hop_edges.flatten().unique()] = True
        sdf_node_2hop_mask[two_hop_edges.flatten().unique()] = True

        # --- Step 1: capture original intermediate embeddings before any deletion hooks ---
        # We hook convs[0] and convs[-1] to capture z1_ori and z2_ori from the unmodified model.
        orig_captured = {}

        def _make_capture_hook(key):
            def hook(module, inp, output):
                orig_captured[key] = output.detach()
                return output
            return hook

        h_orig1 = model.convs[0].register_forward_hook(_make_capture_hook('z1'))
        h_orig2 = model.convs[-1].register_forward_hook(_make_capture_hook('z2'))

        model.eval()
        with torch.no_grad():
            model(self.x, self.edge_index[:, dr_mask])

        h_orig1.remove()
        h_orig2.remove()

        z1_ori = orig_captured['z1']
        z2_ori = orig_captured['z2']

        # --- Step 2: freeze all model parameters ---
        for param in model.parameters():
            param.requires_grad = False

        # --- Step 3: create deletion layers and register them as persistent forward hooks ---
        # Hooks remain after __unlearn__ so that inference also applies the learned deletions.
        hidden_dim = model.hidden_channels[0]
        out_dim = model.convs[-1].out_channels

        deletion1 = DeletionLayer(hidden_dim, sdf_node_1hop_mask).to(self.device)
        deletion2 = DeletionLayer(out_dim, sdf_node_2hop_mask).to(self.device)

        # captured dict is shared with closures so the training loop can read z1/z2 after each forward
        captured = {}

        def _make_deletion_hook(del_layer, key):
            def hook(module, inp, output):
                modified = del_layer(output)
                captured[key] = modified
                return modified
            return hook

        model.convs[0].register_forward_hook(_make_deletion_hook(deletion1, 'z1'))
        model.convs[-1].register_forward_hook(_make_deletion_hook(deletion2, 'z2'))

        # --- Step 4: single optimizer over both deletion layers (matches embdis trainer) ---
        optimizer = torch.optim.Adam(
            list(deletion1.parameters()) + list(deletion2.parameters()),
            lr=self.lr,
        )

        loss_fct = self._get_loss_fct(self.loss_fct_name)
        pos_edge = self.edge_index[:, df_mask]
        n_pos = pos_edge.size(1)

        # --- Step 5: training loop ---
        for epoch in range(self.epochs):
            model.train()
            captured.clear()

            # Forward on SDF edges only; hooks populate captured['z1'] and captured['z2']
            model(self.x, self.edge_index[:, sdf_mask])
            z1, z2 = captured['z1'], captured['z2']

            # Negative edges resampled each epoch (matches original)
            neg_edge = negative_sampling(
                edge_index=self.edge_index,
                num_nodes=num_nodes,
                num_neg_samples=n_pos)

            # Randomness/effectiveness: dot-product decode on z2, MSE between forget-edge
            # logits and negative-edge logits (matches gnndelete_embdis.py:100-101)
            edge_pair = torch.cat([pos_edge, neg_edge], dim=-1)
            df_logits = (z2[edge_pair[0]] * z2[edge_pair[1]]).sum(dim=-1)
            loss_e = loss_fct(df_logits[:n_pos], df_logits[n_pos:])

            # Local causality: preserve embeddings on full SDF node sets, both layers
            loss_l = loss_fct(z1_ori[sdf_node_1hop_mask], z1[sdf_node_1hop_mask]) + \
                     loss_fct(z2_ori[sdf_node_2hop_mask], z2[sdf_node_2hop_mask])

            loss = self.alpha * loss_e + (1 - self.alpha) * loss_l
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            self.info(
                f'GNNDelete epoch {epoch} | '
                f'loss_e={loss_e.item():.4f} loss_l={loss_l.item():.4f}'
            )

        return self.predictor

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get('epochs', 10)
        self.local.config['parameters']['lr'] = self.local.config['parameters'].get('lr', 1e-3)
        self.local.config['parameters']['alpha'] = self.local.config['parameters'].get('alpha', 0.5)
        self.local.config['parameters']['loss_fct'] = self.local.config['parameters'].get('loss_fct', 'mse_mean')
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get('ref_data_forget', 'forget')
