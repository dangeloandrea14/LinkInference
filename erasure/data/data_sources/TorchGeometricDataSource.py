import torch
from .datasource import DataSource
import torch_geometric
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.data.datasets.Dataset import DatasetWrapper 
import numpy as np
from erasure.core.factory_base import get_instance_kvargs
from torch_geometric.transforms import Pad
from torch_geometric.data import Data

from torch.serialization import add_safe_globals
from torch_geometric.data import Data, EdgeAttr, TensorAttr
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

class GeometricWrapper(DatasetWrapper):
    def __init__(self, data, preprocess):
        super().__init__(data,preprocess)
        self.num_nodes = data[0].num_nodes
        

    def __realgetitem__(self, index: int):

        #print("Underlying dataset:", self.data.dataset)
        #print("Subset indices:", self.data.indices)


        sample = self.data[index]

        X = Data(sample.x, sample.edge_index, sample.edge_attr)
        y = sample.y

        y = y.squeeze().long()
     
        return X,y
    
    def get_n_classes(self):
        all_y = torch.cat([data.y for data in self.data], dim=0)
        unique_y = torch.unique(all_y)
        return len(unique_y)
        
        
    def revise_graph_nodes(self, nodes_subset, partitions_dict, remove=False):

        nodes_subset = torch.tensor(nodes_subset, dtype=torch.long)
        nodes_subset = nodes_subset.sort().values

        graph = self.data[0]
        total_nodes = graph.x.size(0)
        edge_index = graph.edge_index

        if remove:
            mask = torch.ones(total_nodes, dtype=torch.bool)
            mask[nodes_subset] = False
            nodes_to_keep = torch.arange(total_nodes)[mask]
        else:
            nodes_to_keep = nodes_subset

        nodes_to_keep = nodes_to_keep.sort().values
        id_map = {old_id.item(): new_id for new_id, old_id in enumerate(nodes_to_keep)}

        edge_mask = (
            torch.isin(edge_index[0], nodes_to_keep) &
            torch.isin(edge_index[1], nodes_to_keep)
        )
        edge_index_sub = edge_index[:, edge_mask].clone()

        edge_index_sub[0] = torch.tensor([id_map[idx.item()] for idx in edge_index_sub[0]])
        edge_index_sub[1] = torch.tensor([id_map[idx.item()] for idx in edge_index_sub[1]])

        x_sub = graph.x[nodes_to_keep]
        y_sub = graph.y[nodes_to_keep]

        data_sub = Data(x=x_sub, edge_index=edge_index_sub, y=y_sub)

        remapped_partitions = {}
        if partitions_dict is not None:
            for key, node_ids in partitions_dict.items():
                if key == 'all':
                    continue
                remapped = []
                for i in node_ids:
                    if i in id_map:
                        remapped.append(id_map[i])
                remapped_partitions[key] = remapped

        return GeometricWrapper([data_sub], self.preprocess), remapped_partitions



    def revise_graph_edges(self, edges_subset, remove=False):

        graph = self.data[0]
        edge_index = graph.edge_index

        edges_subset = set([tuple(edge) for edge in edges_subset]) 

        src = edge_index[0]
        dst = edge_index[1]

        edge_pairs = list(zip(src.tolist(), dst.tolist()))

        if remove:
            mask = torch.tensor(
                [pair not in edges_subset for pair in edge_pairs],
                dtype=torch.bool
            )
        else:
            mask = torch.tensor(
                [pair in edges_subset for pair in edge_pairs],
                dtype=torch.bool
            )

        new_edge_index = edge_index[:, mask]
        new_edge_attr = graph.edge_attr[mask] if graph.edge_attr is not None else None

        data_sub = Data(
            x=graph.x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=graph.y if hasattr(graph, 'y') else None
        )

        return GeometricWrapper([data_sub], self.preprocess)

class TorchGeometricDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)

        self.dataset = None

        add_safe_globals([Data, EdgeAttr, TensorAttr, DataEdgeAttr, GlobalStorage, DataTensorAttr])
    
        self.dataset = get_instance_kvargs(self.local_config['parameters']['datasource']['class'],
                        self.local_config['parameters']['datasource']['parameters'])
        

        self.kwargs = self.local_config['parameters']['datasource']['parameters']

        self.name =  self.local_config['parameters']['datasource']['parameters'].get('name',None)

        print(self.kwargs)


    def get_name(self):
        return self.name


    def create_data(self):

        #Remove empty graphs
        filtered_data_list = [data for data in self.dataset if data.x is not None and data.x.shape[0] > 0]
        filtered_dataset = self.dataset.__class__(**self.kwargs)  
        filtered_dataset.data, filtered_dataset.slices = self.dataset.collate(filtered_data_list)  


        return GeometricWrapper(filtered_dataset, self.preprocess)

    def get_simple_wrapper(self, data):
        return GeometricWrapper(data, self.preprocess)
    

    def check_configuration(self):
        super().check_configuration()

