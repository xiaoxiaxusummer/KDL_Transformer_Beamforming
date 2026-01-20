import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
from torch.utils.data import TensorDataset, DataLoader
import time, math

import os, sys
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(DIRECTORY, '../'))
sys.path.append(PARENT_PATH)
from PASS_setting import generate_data

"""
Normalization on cfg.sigma2 (cfg.beta=1)
"""

torch.cuda.set_device(0)


def proc_data(loc_user, r0, theta, varsigma, cfg):
    n_sample = loc_user.shape[0]
    """ mode 1: [loc_user1_x, loc_user2_x, ...., loc_user1_y, loc_user2_y, ... ]"""
    # obs = np.concatenate((loc_user[:,:,0], loc_user[:,:,1]), axis = 1)
    """ mode 2: [loc_user1_x, loc_user1_y, loc_user2_x, loc_user2_y,.... ]"""
    obs = np.stack((loc_user[:,:,0], loc_user[:,:,1]), axis = 2).reshape((n_sample, -1)) 
    data_mean = torch.tensor(np.array([loc_user[:, :, 0].mean(), loc_user[:,:,1].mean()]))
    data_std = torch.tensor(np.array([loc_user[:, :, 0].std(), loc_user[:,:,1].std()]))
    return torch.tensor(obs), data_mean, data_std

def cal_dist(x, loc_user_x, loc_user_y, loc_PA_y, h_PAA):
    """
    Return antenna-user distance [bs, M, K]
    @ param: x [bs, M]
    @ param: loc_user [bs, K, 3]
    @ param: loc_PA_y [M, ]
    @ param: h_PAA [1,]
    """
    r2 = (x[:,:,None]-loc_user_x[:,None,:])**2 + (loc_PA_y[None,:,None] - loc_user_y[:,None,:])**2 + h_PAA**2 # (bs, M, K)
    # r2 = x[:,:,None] ** 2 - 2 * r0[:,None,:] * theta[:,None,:] *x[:,:,None] + varsigma * r0[:,None,:]**2
    return torch.sqrt(r2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """Compatible with cross attention"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model必须可被num_heads整除"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear propjection with multi-head split
        Q = self._split_heads(self.wq(query))  # (batch_size, num_heads, seq_len, head_dim)
        K = self._split_heads(self.wk(key))
        V = self._split_heads(self.wv(value))

        # Compute attention score
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum & multi-head combination
        attn_output = torch.matmul(attn_weights, V)
        attn_output = self._merge_heads(attn_output)  # (batch_size, seq_len, d_model)
        return self.wo(attn_output)

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)


class EncoderLayer(nn.Module):
    """Process input sequence"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Bi-directional attention
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        # Forward feeding network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class DecoderLayer(nn.Module):
    """Obtain output sequence"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        # Bi-directional self attention
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        # Cross attention (associated with encoder output)
        cross_output = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(cross_output))
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, cfg=None, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super(BidirectionalTransformer, self).__init__()

        self.cfg = cfg
        self.input_dim = 2*cfg.K  # Input sequence length: 2K
        self.output_dim = cfg.M + cfg.N + cfg.K*2
        self.cfg = cfg

        self.max_power = cfg.P_max
        self.sigmoid = nn.Sigmoid()

        # Input processing: mapping each scalar element to a d_model-dimension vector
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Output initialization: A learnable output list of tokens
        self.output_tokens = nn.Parameter(torch.randn(self.output_dim, d_model))  # (M, d_model)

        # Encoder and decoder networks
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Output layer: mapping d_model-dimension vector back to scalars
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, obs):
        # Input: (batch_size, 2K)
        bs = obs.size(0)
        cfg = self.cfg
        src = obs.unsqueeze(-1)  # (bs, 2K, 1)
        src_embedded = self.input_embedding(src)  # (bs, 2K, d_model)
        src_embedded = self.pos_encoder(src_embedded)

        obs = obs.reshape((bs, -1, 2)) # (bs, K, 2)

        # Encoder 
        enc_output = src_embedded
        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output)  # (bs, 2K, d_model)

        # Output sequence initialization
        tgt = self.output_tokens.unsqueeze(0).expand(bs, -1, -1)  # (bs, M, d_model)
        tgt = self.pos_encoder(tgt)

        # Decoder
        dec_output = tgt
        for dec_layer in self.decoder:
            dec_output = dec_layer(dec_output, enc_output)  # (bs, M, d_model)

        # Output sequence computation & projection
        output = self.fc_out(dec_output).squeeze(-1)  # (bs, M)
        output = self.sigmoid(output)
        min_range = cfg.L*cfg.minimum_spacing  # minimum spatial range
        x_end = output[:,0:(cfg.N)]*(cfg.S_PAA -min_range) + min_range # (bs, N)
        PA_spacing_ori = output[:,cfg.N: cfg.N + cfg.M].reshape((bs, cfg.N, cfg.L))  # (bs, N, L)
        pred_lambda = output[:,(cfg.N+cfg.M):(cfg.N+cfg.M+cfg.K)] * 1e12 # Project lambda into the range of [0,1e12] 
        pred_power = output[:,(cfg.N+cfg.M+cfg.K):(cfg.N+cfg.M+cfg.K*2)]
        pred_power = pred_power / pred_power.sum(dim=-1, keepdim=True) * cfg.P_max
        assert (pred_power.sum(-1).mean() - cfg.P_max).abs() < 1e-6


        epsilon = (cfg.minimum_spacing/x_end).unsqueeze(-1) # The minimum value that should be satisfied
        PA_spacing = epsilon + PA_spacing_ori /  (PA_spacing_ori.sum(-1,keepdim=True))*(1-cfg.L * epsilon)
        PA_spacing = PA_spacing * x_end[:,:, None]
        x_pos = torch.cumsum(PA_spacing, dim=-1) # [n_bs, N, L] the position of all antennas
        x_pos = x_pos.reshape((bs, cfg.M))

        loc_PA_y = torch.tensor(cfg.loc_PA[:,1]).to(device, dtype=torch.float64)
        dist = cal_dist(x_pos, obs[:,0:cfg.K,0], obs[:,0:cfg.K,1], loc_PA_y, cfg.h_PAA)
        Theta = (dist + cfg.n_eff * x_pos[:, :, None])
        # Compute real and imag parts
        U_real = np.sqrt(cfg.beta) * torch.cos(Theta) / dist
        U_imag = -np.sqrt(cfg.beta) * torch.sin(Theta) / dist
        assert not torch.isnan(U_real).any() and not torch.isinf(U_real).any()
        assert not torch.isnan(U_imag).any() and not torch.isinf(U_imag).any()
        U = torch.complex(U_real, U_imag)     # Obtain complex numbers
        batch_Gamma = torch.tensor(np.tile(cfg.Gamma,(bs, 1, 1))).to(device, dtype=U.dtype)  # [bs, M, N]
        H_herm = torch.matmul(U.permute([0,2,1]), batch_Gamma) # [bs, K, N]
        H = H_herm.conj().permute([0,2,1]) # [bs, N, K]
        Lambda = torch.diag_embed(pred_lambda).to(torch.complex128) 

        I_mat = torch.eye(cfg.N).unsqueeze(0).repeat(bs, 1, 1).to(device, dtype=torch.float64)
        D_opt = torch.linalg.solve(I_mat + H @ Lambda @ H_herm, H) # [bs, N, K] 
        P_square_root = torch.diag_embed (pred_power.sqrt() / ((torch.abs(D_opt)**2).sum(dim=-2).sqrt() ))# [bs, K, K]
        D = D_opt @ (P_square_root.to(torch.complex128))
        Q = torch.matmul(torch.matmul(U.permute([0,2,1]), batch_Gamma), D)
        signal = torch.abs(Q) ** 2
        effective_gain = torch.diagonal(signal, offset = 0, dim1 = -1, dim2 = -2)
        intf = signal.sum(-1) - effective_gain
        noise_term = intf + cfg.sigma2
        rate = torch.log2(1 + torch.div(effective_gain,noise_term))
        assert not torch.isnan(rate).any() and not torch.isinf(rate).any()
        return x_pos, D, rate


def train(epoch):
    model.train()
    total_loss = 0
    avg_rate = 0
    iter = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if epoch < 10:
            for param in model.parameters():
                param.data += 0.001 * torch.randn_like(param.data)
        x, D, rate = model(data)
        rate = rate.sum(-1).mean()
        loss = torch.neg(rate)
        loss.backward()
        total_loss += loss.item() * data.shape[0]
        avg_rate += rate.item() * data.shape[0]
        iter = iter + data.shape[0]
        optimizer.step()
    return total_loss / iter, avg_rate / iter

def test():
    model.eval()

    total_loss = 0
    avg_rate = 0
    iter = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            start = time.time()
            x, D, rate = model(data)
            rate = rate.sum(-1).mean()
            loss = torch.neg(rate)
            end = time.time()
            print('CGCNet time:', end-start)
            rate = rate.sum(-1).mean()
            avg_rate += rate.item() * data.shape[0]
            total_loss += loss.item() * data.shape[0]
            iter = iter + data.shape[0]
    return total_loss / iter, avg_rate/iter

def perturb_model(model, scale=0.01):
    """Randomly pertub model parameters"""
    with torch.no_grad():
        for param in model.parameters():
            param.add_(scale * torch.randn_like(param))  # Add small perturbations

EMBED_DIM = 256  # Model dimension
NUM_HEADS = 4  #  Number of Heads
NUM_LAYERS = 2  # Number of layers for encoder/decoder networks
DIM_FF = 512  # FFN dimension
dropout = 0.1  # Dropout rate

PATH_SAVE = f"{DIRECTORY}/result/Transformer_Bidirectional_CrossAttn_Perturbed_space_1e12/" 
os.makedirs(PATH_SAVE, exist_ok=True)

# S_PAA = 10
# space_size = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K = 4
N = K 

# for L in [8]:
#     for PMAX_dBM in [16, 10]:
#         for space_size in [5, 10, 15, 20, 25, 30]:
for L in [16, 8]:
    for space_size in [10, 20, 15]:
        for PMAX_dBM in [10, 20, 12, 16, 24, 10, 14, 18, 22]:
            N = K
            M = N * L  # x 的维度
            S_PAA = space_size

            "================== Start new training ====================="
            SEED = 111
            torch.manual_seed(SEED)
            np.random.seed(SEED) 
            S_PAA = space_size

            FILE_NAME = f"{PATH_SAVE}/ckpt_K_{K}_L_{L}_N_{N}_P_{PMAX_dBM}_S_{int(S_PAA)}_Space_{int(space_size)}.pth"
            if os.path.exists(os.path.join(PATH_SAVE, FILE_NAME)):
                continue

            cfg_Train, loc_user_Train, h_tilde_Train, r0_Train, theta_Train, varsigma_Train,_ = \
                generate_data(K=K, L=L, N=N, P_max_dBm = PMAX_dBM, S_PAA=S_PAA, space_size =space_size,  n_sample=30000, saveData=True, testData=False, 
                fileName = f"K_{K}_N_{N}_L_{L}_train.pkl")
            cfg_Train.sigma2 = cfg_Train.sigma2/cfg_Train.beta * cfg_Train.L
            cfg_Train.beta = 1 
            cfg_Train.S_PAA = S_PAA
            train_data_list, data_mean1, data_std1 = proc_data(loc_user_Train, r0_Train, theta_Train, varsigma_Train, cfg_Train)

            cfg_Test, loc_user_Test, h_tilde_Test, r0_Test, theta_Test, varsigma_Test,_ \
                = generate_data(K=K, L=L, N=N, P_max_dBm = PMAX_dBM, S_PAA=S_PAA, space_size =space_size, n_sample=64, saveData=True, testData=True, 
                fileName = f"K_{K}_N_{N}_L_{L}_test.pkl")
            test_data_list, data_mean2, data_std2 = proc_data(loc_user_Test, r0_Test, theta_Test, varsigma_Test, cfg_Test)
            cfg_Test.sigma2 = cfg_Test.sigma2/cfg_Test.beta * cfg_Test.L
            cfg_Test.beta = 1 
            cfg_Test.S_PAA = S_PAA
            
            # Model initialization
            model = BidirectionalTransformer(cfg_Train, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DIM_FF, dropout=0.1).to(device, dtype=torch.float64)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.kaiming_normal_(layer.weight)
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=10,
                num_training_steps=30  # Training steps
            )

            cfg_Train.batch_size = 64
            cfg_Test.batch_size = 64
            train_loader = DataLoader(train_data_list, batch_size=cfg_Train.batch_size, shuffle=True,num_workers=1)
            test_loader = DataLoader(test_data_list, batch_size=cfg_Test.batch_size, shuffle=False, num_workers=1)
            train_losses, test_losses = [], []
            train_rate, test_rate = [], []
            best_rate = 0
            for epoch in range(1, 30):
                loss1, rate1 = train(epoch)
                loss2, rate2 = test()
                print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Train rate: {:.4f}, Test rate: {:.4f}'.format(
                    epoch, loss1, loss2, rate1, rate2))
                scheduler.step()
                if rate1 > best_rate:
                    best_state_dict = model.state_dict()
                train_losses.append(loss1)
                test_losses.append(loss2)
                train_rate.append(rate1)
                test_rate.append(rate2)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_model_state_dict": best_state_dict, 
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_losses,
                "test_loss": test_losses,
                "train_rate": train_rate,
                "test_rate": test_rate,
                "cfg": cfg_Train,
                "EMBED_DIM": EMBED_DIM,
                "NUM_HEADS": NUM_HEADS,
                "NUM_LAYERS": NUM_LAYERS,
            }

            torch.save(checkpoint, os.path.join(PATH_SAVE, FILE_NAME))