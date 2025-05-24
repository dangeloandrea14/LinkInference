from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction

import torch
import torch.nn.functional as F
from torch import nn

from erasure.core.factory_base import get_instance_kvargs

class Noise(nn.Module):
    """
    Noise class to add noise to the model weights. 
    """
    def __init__(self, batch_size, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

    def forward(self):
        return self.noise

class UNSIR(GraphUnlearner):
    def init(self):
        """
        Initializes the UNSIR class with global and local contexts.
        """
        super().init()

        self.epochs = self.local.config['parameters']['epochs'] 
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget']
        self.noise_lr = self.local.config['parameters']['noise_lr']

        self.predictor.optimizer = get_instance_kvargs(
            self.local_config['parameters']['optimizer']['class'],
            {
                'params': self.predictor.model.parameters(), 
                **self.local_config['parameters']['optimizer']['parameters']
            }
        )

        self.sample_type = self.local.config['parameters']['sample_type']

    def __unlearn__(self):
        """
        UNSIR adapted for graph-based node classification.
        """
        self.info(f'Starting UNSIR with {self.epochs} epochs for the impair phase')

        self.forget_set = torch.tensor(self.dataset.partitions[self.ref_data_forget]).to(self.device)
        self.retain_set = torch.tensor(self.dataset.partitions[self.ref_data_retain]).to(self.device)

        num_nodes = self.x.size(0)
        all_nodes = torch.arange(num_nodes)

        if self.removal_type == 'edge':
            self.forget_set = self.infected_nodes(self.forget_set, self.hops)
            self.retain_set = [node for node in all_nodes if node not in self.forget_set]

        x = self.x.to(self.device)
        edge_index = self.edge_index.to(self.device)
        labels = self.labels.to(self.device)

        for epoch in range(self.epochs):
            running_loss = 0

            noise = Noise(len(self.forget_set), x.size(1)).to(self.device)

            print(f"x[forget_set] shape: {x[self.forget_set].shape}")
            print(f"noise() shape: {noise().shape}")
            noise_optimizer = torch.optim.Adam(noise.parameters(), lr=self.noise_lr)

            for _ in range(5):
                x_noised = x.clone()
                x_noised[self.forget_set] = x[self.forget_set] + noise()

                outputs = self.predictor.model(x_noised, edge_index)[self.forget_set]

                with torch.no_grad():
                    target_logits = self.predictor.model(x, edge_index)[self.forget_set]

                loss_noise = -F.mse_loss(outputs, target_logits)  
                noise_optimizer.zero_grad()
                loss_noise.backward(retain_graph=True)
                noise_optimizer.step()

            with torch.no_grad():
                noise_tensor = torch.clamp(noise(), -1, 1)
                x_noised = x.clone()
                x_noised[self.forget_set] = x[self.forget_set] + noise_tensor

            outputs_noised = self.predictor.model(x_noised, edge_index)[self.retain_set]
            loss_1 = self.predictor.loss_fn(outputs_noised, labels[self.retain_set])

            outputs_clean = self.predictor.model(x, edge_index)[self.retain_set]
            loss_2 = self.predictor.loss_fn(outputs_clean, labels[self.retain_set])

            joint_loss = loss_1 + loss_2

            self.predictor.optimizer.zero_grad()
            joint_loss.backward()
            self.predictor.optimizer.step()

            running_loss += joint_loss.item() * len(self.retain_set)
            average_train_loss = running_loss / (len(self.retain_set) * 1)  # 1 epoch

            self.info(f'UNSIR - Epoch {epoch+1}/{self.epochs} - Avg Loss: {average_train_loss:.4f}')

        return self.predictor

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 1)
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain')
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')
        self.local.config['parameters']['noise_lr'] = self.local.config['parameters'].get("noise_lr", 0.01)
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class': 'torch.optim.Adam', 'parameters': {}})
        self.local.config['parameters']['sample_type'] = self.local.config['parameters'].get("sample_type", 'default')
