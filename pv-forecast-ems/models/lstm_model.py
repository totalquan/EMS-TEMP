"""
模型开发层 - LSTM时序预测模型
基于PyTorch实现，支持多变量输入、多步预测
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from models.base_model import BasePVModel
from config.settings import settings


class LSTMNet(nn.Module):
    """LSTM网络结构"""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


class LSTMModel(BasePVModel):
    """LSTM光伏预测模型"""

    def __init__(self, model_name: str = "lstm_pv_default",
                 seq_length: int = None,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 epochs: int = 100,
                 early_stopping_patience: int = 10):
        super().__init__(model_name=model_name, model_type="lstm")

        self.seq_length = seq_length or settings.forecast.seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[LSTMNet] = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        logger.info(f"LSTM模型初始化 | device={self.device} | seq_length={self.seq_length}")

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将平面数据转换为LSTM序列数据"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
            y_seq.append(y[i + self.seq_length])
        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """训练LSTM模型"""
        start_time = time.time()
        logger.info(f"开始训练LSTM模型 | 训练样本={len(X_train)}")

        input_size = X_train.shape[1] if len(X_train.shape) > 1 else 1
        self.feature_cols = kwargs.get("feature_cols", [])

        # 转换为序列
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        logger.info(f"序列数据构建完成 | 训练序列数={len(X_train_seq)}")

        # 转为Tensor
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 验证集
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 构建网络
        self.net = LSTMNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # 训练循环
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.net(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # 验证
            avg_val_loss = avg_train_loss
            if val_loader:
                self.net.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.net(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        val_batches += 1
                avg_val_loss = val_loss / max(val_batches, 1)
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)

            # 早停
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} | "
                           f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f}")

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"早停触发 | epoch={epoch+1} | best_val_loss={best_val_loss:.6f}")
                break

        # 恢复最佳模型
        if best_state:
            self.net.load_state_dict(best_state)

        self.is_trained = True
        training_time = time.time() - start_time

        self.training_info.update({
            "training_samples": len(X_train_seq),
            "training_time_sec": round(training_time, 2),
            "epochs_trained": epoch + 1,
            "best_val_loss": round(best_val_loss, 6),
            "final_train_loss": round(train_losses[-1], 6),
            "hyperparameters": {
                "seq_length": self.seq_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
            },
        })

        logger.info(f"LSTM训练完成 | 用时={training_time:.1f}s | best_val_loss={best_val_loss:.6f}")
        return self.training_info

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained or self.net is None:
            raise RuntimeError("模型未训练，无法预测")

        self.net.eval()

        # 处理输入形状
        if len(X.shape) == 2:
            # 需要构造序列
            if len(X) < self.seq_length:
                # 数据不足，用最后一个值填充
                pad = np.repeat(X[-1:], self.seq_length - len(X), axis=0)
                X = np.vstack([pad, X])
            X_seq = X[-self.seq_length:][np.newaxis, :, :]
        elif len(X.shape) == 3:
            X_seq = X
        else:
            raise ValueError(f"不支持的输入形状: {X.shape}")

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            outputs = self.net(X_tensor)
            predictions = outputs.cpu().numpy().flatten()

        return predictions

    def save(self, save_dir: str) -> str:
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{self.model_name}.pth")

        torch.save({
            "model_state_dict": self.net.state_dict() if self.net else None,
            "input_size": self.net.lstm.input_size if self.net else None,
            "seq_length": self.seq_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }, model_path)

        self._save_metadata(save_dir)
        logger.info(f"LSTM模型已保存: {model_path}")
        return model_path

    def load(self, model_path: str) -> bool:
        """加载模型"""
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            self.seq_length = checkpoint.get("seq_length", self.seq_length)
            self.hidden_size = checkpoint.get("hidden_size", self.hidden_size)
            self.num_layers = checkpoint.get("num_layers", self.num_layers)
            self.dropout = checkpoint.get("dropout", self.dropout)
            input_size = checkpoint.get("input_size", 8)

            self.net = LSTMNet(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)

            if checkpoint.get("model_state_dict"):
                self.net.load_state_dict(checkpoint["model_state_dict"])

            self.is_trained = True

            # 加载元数据
            save_dir = os.path.dirname(model_path)
            self._load_metadata(save_dir)

            logger.info(f"LSTM模型已加载: {model_path}")
            return True

        except Exception as e:
            logger.error(f"加载LSTM模型失败: {e}")
            return False
