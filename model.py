import torch
from torch.utils.data import Dataset, Sampler
from torch.nn import (
    Module,
    Sequential,
    Linear,
    ReLU,
    Sigmoid,
    Embedding,
    ModuleDict,
    ModuleList,
    Dropout,
    BatchNorm1d,
    Identity,
    functional,
)

import torch_sparse
import torch_scatter
import torch_geometric as tg
from torch_geometric.data import Data, Batch

# %% [markdown]
# ## Loading as a graph

# %%
class DecoyDataset(Dataset):
    def __init__(self, pth_paths, transforms=None):
        super().__init__()

        self.targets_by_target_id: Dict[str, Data] = {}
        self.targets_by_casp_ed_and_target_id: Dict[Tuple[int, str], Data] = {}
        self.decoys_by_target_id_and_decoy_id: Dict[Tuple[str, str], Data] = {}
        self.decoys_by_casp_ed_and_target_id_and_decoy_id: Dict[
            Tuple[int, str, str], Data
        ] = {}

        logger.info("Starting to load graphs")
        for p in pth_paths:
            target = torch.load(p)
            casp_ed = target["casp_ed"]
            target_id = target["target_id"]

            self.targets_by_target_id[target_id] = target
            self.targets_by_casp_ed_and_target_id[(casp_ed, target_id)] = target

            for decoy in target["graphs"]:
                decoy_id = decoy.decoy_id
                self.add_target_feats_to_decoy(target, decoy)
                self.decoys_by_target_id_and_decoy_id[(target_id, decoy_id)] = decoy
                self.decoys_by_casp_ed_and_target_id_and_decoy_id[
                    (casp_ed, target_id, decoy_id)
                ] = decoy

        if len(self) == 0:
            logger.warning("Empty dataset!")
        else:
            logger.info(f"Done loading {len(self)} graphs")
        self.transforms = transforms

    @staticmethod
    def add_target_feats_to_decoy(target: dict, decoy: Data):
        decoy.casp_ed = target["casp_ed"]
        decoy.target_id = target["target_id"]
        decoy.n_nodes = decoy.num_nodes
        decoy.n_edges = decoy.num_edges
        decoy.msa_feats = target["msa_feats"]
        decoy.aa = target["sequence"]

        if decoy.num_nodes == 0:
            # If a graph has 0 nodes and it's put last in the the batch formed by
            # Batch.from_data_list(...) it will cause a miscount in batch.num_graphs
            logger.warning(
                f"Found graph with 0 nodes: {decoy.casp_ed}/{decoy.target_id}/{decoy.decoy_id}"
            )

    def __getitem__(self, item):
        if isinstance(item, int):
            item = self.keys[item]
        *casp_ed, target_id, decoy_id = item
        graph = self.decoys_by_target_id_and_decoy_id[(target_id, decoy_id)]
        if self.transforms is not None:
            graph = self.transforms(graph.clone())
        return graph

    def __len__(self):
        return len(self.decoys_by_casp_ed_and_target_id_and_decoy_id)

    @functools.cached_property
    def casp_editions(self) -> Tuple[int]:
        return tuple(set(k[0] for k in self.targets_by_casp_ed_and_target_id.keys()))

    @functools.cached_property
    def target_ids(self) -> Tuple[str]:
        return tuple(self.targets_by_target_id.keys())

    @functools.cached_property
    def keys(self) -> Tuple[Tuple[int, str, str], ...]:
        return tuple(self.decoys_by_casp_ed_and_target_id_and_decoy_id.keys())


pth_files = Path().glob(f"CASP13/processed/*.pth")
ds = DecoyDataset(pth_files)

print(ds[0])
print(ds[1])
print(ds[2230])
print(ds[2231])

count = 0
for batch in tg.data.DataLoader(ds, batch_size=128):
    batch.debug()
    count += batch.num_graphs
print(count)


# %% [markdown]
# ## Graph Network

# %%
def round_to_pow2(value):
    return np.exp2(np.round(np.log2(value))).astype(int)


def layer_sizes_linear(in_feats, out_feats, layers, round_pow2=False):
    sizes = np.linspace(in_feats, out_feats, layers + 1).round().astype(np.int)
    if round_pow2:
        sizes[1:-1] = round_to_pow2(sizes[1:-1])
    return sizes.tolist()


def layer_sizes_exp2(in_feats, out_feats, layers, round_pow2=False):
    sizes = (
        np.logspace(np.log2(in_feats), np.log2(out_feats), layers + 1, base=2)
        .round()
        .astype(np.int)
    )
    if round_pow2:
        sizes[1:-1] = round_to_pow2(sizes[1:-1])
    return sizes.tolist()


# %%
class qamodel(Module):
    def __init__(self):
        super().__init__()

        # Configuration
        aa_embedding_dim = 64
        ss_embedding_dim = 64
        sep_embedding_dim = 64
        rbf_num_bases = 16

        enc_out_node_feats = 128
        enc_out_edge_feats = 64

        mp_layers = 2
        mp_dropout = 0.2
        mp_batch_norm = False
        mp_in_edge_feats = enc_out_edge_feats + sep_embedding_dim + rbf_num_bases
        mp_in_node_feats = enc_out_node_feats + aa_embedding_dim + ss_embedding_dim
        mp_in_global_feats = 512
        mp_out_edge_feats = 16
        mp_out_node_feats = 64
        mp_out_global_feats = 32
        mp_edge_feats = layer_sizes_exp2(
            mp_in_edge_feats, mp_out_edge_feats, mp_layers, round_pow2=True
        )
        mp_node_feats = layer_sizes_exp2(
            mp_in_node_feats, mp_out_node_feats, mp_layers, round_pow2=True
        )
        mp_global_feats = layer_sizes_exp2(
            mp_in_global_feats, mp_out_global_feats, mp_layers, round_pow2=True
        )
        mp_sizes = zip(mp_edge_feats, mp_node_feats, [0] + mp_global_feats[1:])

        self.readout_concat = True

        # Embeddings (aa type, dssp classification, separation, distance)
        self.embeddings = ModuleDict(
            {
                "amino_acid": Embedding(
                    num_embeddings=20, embedding_dim=aa_embedding_dim
                ),
                "secondary_structure": Embedding(
                    num_embeddings=9, embedding_dim=ss_embedding_dim
                ),
                "separation": SeparationEmbedding(
                    bins=(1, 2, 3, 4, 5, 10, 15), embedding_dim=sep_embedding_dim
                ),
                "distance_rbf": RbfDistanceEncoding(
                    min_dist=0, max_dist=20, num_bases=rbf_num_bases
                ),
            }
        )

        # Encoder (dssp features on the nodes and geometric features on the edges)
        self.encoder = Encoder(
            out_edge_feats=enc_out_edge_feats, out_node_feats=enc_out_node_feats
        )

        # GraphTransformerLayer
        self.graph_transformer_layer = GraphTransformerLayer(
            in_node_dim = mp_in_node_feats,
            in_edge_dim = mp_in_edge_feats,
            out_dim = mp_in_node_feats,
            out_edge_dim = mp_in_edge_feats,
            num_heads = 4,
            dropout = 0.0,
            residual = True,
            batch_norm = True,
            use_bias = False,
        )

        # Message passing
        self.message_passing = ModuleList()
        in_e, in_n, in_g = next(mp_sizes)
        for out_e, out_n, out_g in mp_sizes:
            mp = MessagePassing(
                in_edge_feats=in_e,
                in_node_feats=in_n,
                in_global_feats=in_g,
                out_edge_feats=out_e,
                out_node_feats=out_n,
                out_global_feats=out_g,
                dropout=mp_dropout,
                batch_norm=mp_batch_norm,
            )
            self.message_passing.append(mp)
            in_e, in_n, in_g = out_e, out_n, out_g

        # Readout
        if self.readout_concat:
            in_n += mp_in_node_feats
        self.readout = Readout(in_n, in_g)

    @staticmethod
    def prepare(graphs: Batch) -> Tuple[torch.Tensor, ...]: # prepare一个batch的相关数据作为模型的输入，而不是以一整个graph作为输入
        aa = graphs.aa
        msa_feats = graphs.msa_feats
        x = graphs.x
        edge_index = graphs.edge_index
        edge_attr = graphs.edge_attr
        secondary_structure = graphs.secondary_structure
        batch = graphs.batch # size为该batch里的num_nodes，值为该位置对应的node属于哪个图

        return aa, msa_feats, x, edge_index, edge_attr, secondary_structure, batch

    def forward(
        self, aa, msa_feats, x, edge_index, edge_attr, secondary_structure, batch
    ):
        # Embeddings (aa type, dssp classification, separation, distance)
        aa = self.embeddings.amino_acid(aa.long())
        ss = self.embeddings.secondary_structure(secondary_structure.long())
        sep = self.embeddings.separation(edge_index)
        rbf = self.embeddings.distance_rbf(edge_attr[:, 0])

        # Encoder (dssp features on the nodes and geometric features on the edges)
        x, edge_attr = self.encoder(x, msa_feats, edge_attr)

        # Message passing
        x = x_mp = torch.cat((aa, x, ss), dim=1)
        edge_attr = torch.cat((edge_attr, sep, rbf), dim=1)

        # 在这里加一个GraphTransformerLayer试一下
        x, edge_attr = self.graph_transformer_layer(edge_index, x, edge_attr, batch)

        num_graphs = batch[-1].item() + 1
        u = torch.empty(num_graphs, 0, dtype=torch.float, device=x.device)
        for mp in self.message_passing:
            x, edge_attr, edge_index, u, batch = mp(x, edge_attr, edge_index, u, batch)

        # Readout
        if self.readout_concat:
            x = torch.cat((x, x_mp), dim=1)
        x, u = self.readout(x, u)
        return x, u


class SeparationEmbedding(Module):
    def __init__(self, embedding_dim, bins: tuple):
        super().__init__()
        self.bins = bins
        self.emb = Embedding(num_embeddings=len(bins) + 1, embedding_dim=embedding_dim)

    @torch.jit.ignore
    def _sep_to_code(self, separation):
        codes = np.digitize(separation.abs().cpu().numpy(), bins=self.bins, right=True)
        codes = torch.from_numpy(codes).to(separation.device)
        return codes

    def forward(self, edge_index):
        separation = edge_index[0] - edge_index[1]
        codes = self._sep_to_code(separation)
        embeddings = self.emb(codes)
        return embeddings


class RbfDistanceEncoding(Module):
    def __init__(self, min_dist: float, max_dist: float, num_bases: int):
        super().__init__()
        if not 0 <= min_dist < max_dist:
            raise ValueError(
                f"Invalid RBF centers: 0 <= {min_dist} < {max_dist} is False"
            )
        if num_bases < 0:
            raise ValueError(f"Invalid RBF size: 0 < {num_bases} is False")
        self.register_buffer(
            "rbf_centers", torch.linspace(min_dist, max_dist, steps=num_bases)
        )

    def forward(self, distances):
        # assert distances.ndim == 1
        # Distances are encoded using a equally spaced RBF kernels with unit variance 使用具有单位方差的等距RBF核对距离进行编码
        rbf = torch.exp(-((distances[:, None] - self.rbf_centers[None, :]) ** 2))
        return rbf

    def extra_repr(self):
        return f"bases={len(self.rbf_centers)}"


class Encoder(Module):
    def __init__(self, out_edge_feats, out_node_feats):
        super().__init__()
        self.node_encoder = Sequential(
            Linear(3 + 21, out_node_feats // 2),
            ReLU(),
            Linear(out_node_feats // 2, out_node_feats),
            ReLU(),
        )
        self.edge_encoder = Sequential(
            Linear(4, out_edge_feats // 2),
            ReLU(),
            Linear(out_edge_feats // 2, out_edge_feats),
            ReLU(),
        )

    def forward(self, x, msa_feats, edge_attr):
        x = self.node_encoder(torch.cat((x, msa_feats), dim=1))
        edge_attr = self.edge_encoder(edge_attr)
        return x, edge_attr


class MessagePassing(Module):
    def __init__(
        self,
        in_edge_feats: int,
        in_node_feats: int,
        in_global_feats: int,
        out_edge_feats: int,
        out_node_feats: int,
        out_global_feats: int,
        batch_norm: bool,
        dropout: float,
    ):
        super().__init__()
        in_feats = in_node_feats + in_edge_feats + in_global_feats
        self.edge_fn = Sequential(
            Linear(in_feats, out_edge_feats),
            Dropout(p=dropout) if dropout > 0 else Identity(),
            BatchNorm1d(out_edge_feats) if batch_norm else Identity(),
            ReLU(),
        )
        in_feats = in_node_feats + out_edge_feats + in_global_feats
        self.node_fn = Sequential(
            Linear(in_feats, out_node_feats),
            Dropout(p=dropout) if dropout > 0 else Identity(),
            BatchNorm1d(out_node_feats) if batch_norm else Identity(),
            ReLU(),
        )
        in_feats = out_node_feats + out_edge_feats + in_global_feats
        self.global_fn = Sequential(
            Linear(in_feats, out_global_feats),
            Dropout(p=dropout) if dropout > 0 else Identity(),
            BatchNorm1d(out_global_feats) if batch_norm else Identity(),
            ReLU(),
        )

    def forward(self, x, edge_attr, edge_index, u, batch):
        x_src = x[edge_index[0]] # 初始size (num_edges,3)
        u_src = u[batch[edge_index[0]]] # 初始size为(num_edges,4),batch这个tensor里的值是该位置对应的node属于哪个图
        edge_attr = torch.cat((x_src, edge_attr, u_src), dim=1) # 初始size (num_edges,3+4+4)
        edge_attr = self.edge_fn(edge_attr) # 更新edge_attr

        msg_to_node = torch_scatter.scatter(
            edge_attr, edge_index[1], dim=0, dim_size=x.shape[0], reduce="mean"
        ) # 将msg以mean的方式聚合到目标节点
        u_to_node = u[batch]
        x = torch.cat((x, msg_to_node, u_to_node), dim=1) # concatenate
        x = self.node_fn(x) # 更新x

        edge_global = torch_scatter.scatter(
            edge_attr, batch[edge_index[0]], dim=0, dim_size=u.shape[0], reduce="mean"
        ) # 以mean的方式聚合每个图的边特征
        x_global = torch_scatter.scatter(
            x, batch, dim=0, dim_size=u.shape[0], reduce="mean"
        ) # 以mean的方式聚合每个图的节点特征
        u = torch.cat((edge_global, x_global, u), dim=1) # concatenate
        u = self.global_fn(u) # 更新u

        return x, edge_attr, edge_index, u, batch


class MultiHeadAttentionLayer(Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        # edge

        if use_bias:
            self.Q = Linear(in_node_dim, out_dim * num_heads, bias=True)
            self.K = Linear(in_node_dim, out_dim * num_heads, bias=True)
            self.V = Linear(in_node_dim, out_dim * num_heads, bias=True)
            self.proj_e = Linear(in_edge_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = Linear(in_node_dim, out_dim * num_heads, bias=False)
            self.K = Linear(in_node_dim, out_dim * num_heads, bias=False)
            self.V = Linear(in_node_dim, out_dim * num_heads, bias=False)
            self.proj_e = Linear(in_edge_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, edge_index, K_h, Q_h, V_h, proj_e):

        K_h = K_h[edge_index[0]]
        Q_h = Q_h[edge_index[1]]
        V_h = V_h[edge_index[0]]
        proj_e = proj_e
        # 计算 attention score
        atten_score = K_h * Q_h
        # scaling
        atten_score = atten_score / np.sqrt(self.out_dim)

        atten_score = atten_score + proj_e

        e_out = atten_score

        atten_score = torch.exp((atten_score.sum(-1, keepdim=True)).clamp(-5, 5))

        V_h = torch.mul(V_h, atten_score)
        wV = torch_scatter.scatter(V_h, edge_index[0], dim=0, reduce="sum")  # 相同目标节点的V_h聚合起来
        z = torch_scatter.scatter(atten_score, edge_index[0], dim=0, reduce="mean")  # 相同目标节点的atten_score聚合起来

        return wV, z, e_out

    def forward(self, edge_index, h, e):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        edge_index = edge_index

        # view成[num_nodes, num_heads, feat_dim]来得到multi-head attention的投影

        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)
        proj_e = proj_e.view(-1, self.num_heads, self.out_dim)

        wV, z, e_out = self.propagate_attention(edge_index, K_h, Q_h, V_h, proj_e)

        h_out = wV / (z + torch.full_like(z, 1e-6))  # adding eps to all values here

        return h_out, e_out  # size分别为(num_nodes, num_heads, out_dim), (num_edges, num_heads, out_dim)


class GraphTransformerLayer(Module):

    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        out_dim: int,
        out_edge_dim: int,
        num_heads: int,
        dropout: float,
        residual: bool,
        batch_norm: bool,
        use_bias: bool,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        self.residual = residual

        self.attention = MultiHeadAttentionLayer(in_node_dim, in_edge_dim, out_dim // num_heads, num_heads, use_bias)

        self.O_h = Linear(out_dim, out_dim)
        self.O_e = Linear(out_dim, out_edge_dim)

        self.batch_norm1_h = BatchNorm1d(out_dim)
        self.batch_norm1_e = BatchNorm1d(out_edge_dim)

        # FFN for h
        self.FFN_h_layer1 = Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = Linear(out_dim * 2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = Linear(out_edge_dim, out_edge_dim * 2)
        self.FFN_e_layer2 = Linear(out_edge_dim * 2, out_edge_dim)

        self.batch_norm2_h = BatchNorm1d(out_dim)
        self.batch_norm2_e = BatchNorm1d(out_edge_dim)

    def forward(self, edge_index, h, e, batch):
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(edge_index, h, e)

        h = h_attn_out.view(-1, self.out_dim)
        e = e_attn_out.view(-1, self.out_dim)

        h = functional.dropout(h, self.dropout, training=self.training)
        e = functional.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e) # 变换e的dim为原来的dim

        if self.residual:
            h = h_in1 + h  # residual connection
            e = e_in1 + e  # residual connection

        h = self.batch_norm1_h(h)
        e = self.batch_norm1_e(e)

        h_in2 = h  # for second residual connection
        e_in2 = e  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = functional.relu(h)
        h = functional.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = functional.relu(e)
        e = functional.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # residual connection
            e = e_in2 + e  # residual connection

        h = self.batch_norm2_h(h)
        e = self.batch_norm2_e(e)

        return h, e


class Readout(Module):
    def __init__(self, in_node_feats, in_global_feats):
        super().__init__()
        self.node_fn = Sequential(Linear(in_node_feats, 2), Sigmoid())
        self.global_fn = Sequential(Linear(in_global_feats, 5), Sigmoid())

    def forward(self, x, u):
        x = self.node_fn(x)
        u = self.global_fn(u)
        return x, u
