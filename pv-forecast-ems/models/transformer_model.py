"""
模型开发层 - Transformer时序预测模型
基于PyTorch实现，使用自注意力机制捕捉长程依赖
"""
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from models.base_model import BasePVModel
from config.settings import settings


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    """Transformer预测网络"""
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 取最后一个时间步
        return self.output_proj(x)


class TransformerModel(BasePVModel):
    """Transformer光伏预测模型"""

    def __init__(self, model_name: str = "transformer_pv_default",
                 seq_length: int = None, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dropout: float = 0.1,
                 learning_rate: float = 0.0005, batch_size: int = 64,
                 epochs: int = 80, early_stopping_patience: int = 10):
        super().__init__(model_name=model_name, model_type="transformer")
        self.seq_length = seq_length or settings.forecast.seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[TransformerNet] = None
        self.criterion = nn.MSELoss()

    def _create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
            y_seq.append(y[i + self.seq_length])
        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        start_time = time.time()
        logger.info(f"开始训练Transformer模型 | 训练样本={len(X_train)}")
        input_size = X_train.shape[1] if len(X_train.shape) > 1 else 1
        self.feature_cols = kwargs.get("feature_cols", [])

        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        X_train_t = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_t = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                   batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            X_val_t = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_t = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
            val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                                    batch_size=self.batch_size, shuffle=False)

        self.net = TransformerNet(input_size, self.d_model, self.nhead,
                                   self.num_layers, dropout=self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = self.criterion(self.net(batch_X), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_train_loss = epoch_loss / len(train_loader)
            scheduler.step()

            avg_val_loss = avg_train_loss
            if val_loader:
                self.net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        val_loss += self.criterion(self.net(batch_X), batch_y).item()
                avg_val_loss = val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} | train={avg_train_loss:.6f} | val={avg_val_loss:.6f}")
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"早停 | epoch={epoch+1}")
                break

        if best_state:
            self.net.load_state_dict(best_state)
        self.is_trained = True
        self.training_info.update({
            "training_samples": len(X_train_seq),
            "training_time_sec": round(time.time() - start_time, 2),
            "best_val_loss": round(best_val_loss, 6),
            "hyperparameters": {"d_model": self.d_model, "nhead": self.nhead,
                                "num_layers": self.num_layers, "dropout": self.dropout},
        })
        return self.training_info

    def predict(self, X):
        if not self.is_trained or self.net is None:
            raise RuntimeError("模型未训练")
        self.net.eval()
        if len(X.shape) == 2:
            if len(X) < self.seq_length:
                pad = np.repeat(X[-1:], self.seq_length - len(X), axis=0)
                X = np.vstack([pad, X])
            X = X[-self.seq_length:][np.newaxis, :, :]
        with torch.no_grad():
            return self.net(torch.FloatTensor(X).to(self.device)).cpu().numpy().flatten()

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.model_name}.pth")
        torch.save({
            "model_state_dict": self.net.state_dict() if self.net else None,
            "input_size": self.net.input_proj.in_features if self.net else None,
            "seq_length": self.seq_length, "d_model": self.d_model,
            "nhead": self.nhead, "num_layers": self.num_layers, "dropout": self.dropout,
        }, path)
        self._save_metadata(save_dir)
        return path

    def load(self, model_path):
        if not os.path.exists(model_path):
            return False
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            self.seq_length = ckpt.get("seq_length", self.seq_length)
            self.net = TransformerNet(ckpt.get("input_size", 8), ckpt.get("d_model", 128),
                                       ckpt.get("nhead", 8), ckpt.get("num_layers", 3),
                                       ckpt.get("dropout", 0.1)).to(self.device)
            if ckpt.get("model_state_dict"):
                self.net.load_state_dict(ckpt["model_state_dict"])
            self.is_trained = True
            self._load_metadata(os.path.dirname(model_path))
            return True
        except Exception as e:
            logger.error(f"加载Transformer失败: {e}")
            return False
