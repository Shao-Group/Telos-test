#!/usr/bin/env python3
"""
Ultra-simple site-level CNN for TSS/TES prediction.
Stripped down to bare minimum for debugging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict
import os
import warnings
warnings.filterwarnings("ignore")

class SiteLevelCNN(nn.Module):
    """Proper CNN for TSS/TES prediction from soft-clipped sequences."""
    
    def __init__(self, max_seq_length=50, embedding_dim=32):
        super(SiteLevelCNN, self).__init__()
        
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        
        # Proper CNN layers for motif detection
        self.conv1 = nn.Conv1d(5, 32, kernel_size=3, padding=1)  # 5 -> 32 channels
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)  # 32 -> 64 channels
        self.conv3 = nn.Conv1d(64, 32, kernel_size=7, padding=3)  # 64 -> 32 channels
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(0.3)
        
        # Global pooling to get fixed-size representation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Sequence embedding layer (combines avg and max pooling)
        self.seq_embedding = nn.Sequential(
            nn.Linear(64, embedding_dim),  # 32*2 (avg+max) -> embedding_dim
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, sequences_tensor, return_embeddings=False):
        """
        Forward pass: sequences -> CNN -> embeddings -> site aggregation -> prediction
        
        Args:
            sequences_tensor: (num_sequences_for_one_site, 5, seq_length)
            return_embeddings: Return site embedding instead of prediction
        """
        if sequences_tensor.shape[0] == 0:
            if return_embeddings:
                return torch.zeros(self.embedding_dim, device=sequences_tensor.device)
            else:
                return torch.tensor(0.5, device=sequences_tensor.device)
        
        # CNN feature extraction
        x = sequences_tensor  # (num_sequences, 5, seq_length)
        
        # First conv layer
        x = self.relu(self.bn1(self.conv1(x)))  # (num_sequences, 32, seq_length)
        x = self.dropout(x)
        
        # Second conv layer
        x = self.relu(self.bn2(self.conv2(x)))  # (num_sequences, 64, seq_length)
        x = self.dropout(x)
        
        # Third conv layer
        x = self.relu(self.bn3(self.conv3(x)))  # (num_sequences, 32, seq_length)
        
        # Global pooling (both avg and max)
        x_avg = self.global_avg_pool(x).squeeze(-1)  # (num_sequences, 32)
        x_max = self.global_max_pool(x).squeeze(-1)  # (num_sequences, 32)
        x_pooled = torch.cat([x_avg, x_max], dim=1)  # (num_sequences, 64)
        
        # Sequence embeddings
        seq_embeddings = self.seq_embedding(x_pooled)  # (num_sequences, embedding_dim)
        
        # Site-level aggregation (mean + attention-like weighting)
        # Simple attention mechanism
        attention_weights = torch.softmax(seq_embeddings.sum(dim=1), dim=0)  # (num_sequences,)
        site_embedding = (seq_embeddings * attention_weights.unsqueeze(1)).sum(dim=0)  # (embedding_dim,)
        
        if return_embeddings:
            return site_embedding
        
        # Classify this site
        output = self.classifier(site_embedding.unsqueeze(0))  # (1, 1)
        
        if self.training:
            return output.squeeze()  # scalar
        else:
            return self.sigmoid(output).squeeze()  # scalar


class SiteCNNEmbedder:
    """Simple embedder for inference."""
    
    def __init__(self, model_path=None, max_seq_length=50, embedding_dim=32, clip_type='start'):
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.clip_type = clip_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SiteLevelCNN(max_seq_length, embedding_dim)
        self.model.to(self.device)
        self.model.eval()
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded {clip_type} model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load {clip_type} model: {e}")
        else:
            print(f"Using random {clip_type} model")
    
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """One-hot encode DNA sequence."""
        mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        
        # Clean and pad sequence
        seq = ''.join([base.upper() for base in sequence if base.upper() in 'ATGCN'])
        if not seq:
            seq = 'N'
        
        # Pad/truncate - for soft clips, center the sequence if it's shorter
        if len(seq) > self.max_seq_length:
            # If too long, take the middle part
            start = (len(seq) - self.max_seq_length) // 2
            seq = seq[start:start + self.max_seq_length]
        else:
            # If too short, pad equally on both sides
            pad_total = self.max_seq_length - len(seq)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            seq = 'N' * pad_left + seq + 'N' * pad_right
        
        # One-hot encode
        encoded = np.zeros((5, len(seq)))
        for i, base in enumerate(seq):
            if base in mapping:
                encoded[mapping[base], i] = 1
            else:
                encoded[4, i] = 1
                
        return encoded
    
    def predict_site(self, site_sequences: List[str]) -> float:
        """Predict TSS/TES probability for a site."""
        if not site_sequences:
            return 0.0
        
        # Encode sequences
        encoded_sequences = []
        for seq in site_sequences:
            if seq.strip():
                encoded_seq = self._encode_sequence(seq)
                encoded_sequences.append(torch.FloatTensor(encoded_seq).to(self.device))
        
        if not encoded_sequences:
            return 0.0
        
        # Stack and predict
        sequences_tensor = torch.stack(encoded_sequences)
        with torch.no_grad():
            prediction = self.model(sequences_tensor, return_embeddings=False)
            
        return float(prediction.cpu().numpy())
    
    def embed_site(self, site_sequences: List[str]) -> np.ndarray:
        """Generate embedding for a site."""
        if not site_sequences:
            return np.zeros(self.embedding_dim)
        
        # Encode sequences
        encoded_sequences = []
        for seq in site_sequences:
            if seq.strip():
                encoded_seq = self._encode_sequence(seq)
                encoded_sequences.append(torch.FloatTensor(encoded_seq).to(self.device))
        
        if not encoded_sequences:
            return np.zeros(self.embedding_dim)
        
        # Stack and embed
        sequences_tensor = torch.stack(encoded_sequences)
        with torch.no_grad():
            embedding = self.model(sequences_tensor, return_embeddings=True)
            
        return embedding.cpu().numpy()


class SiteCNNTrainer:
    """Simple trainer."""
    
    def __init__(self, max_seq_length=50, embedding_dim=32, clip_type='start'):
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.clip_type = clip_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = SiteLevelCNN(max_seq_length, embedding_dim)
        self.model.to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)  # Lower learning rate with regularization
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode single sequence."""
        mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        
        # Clean sequence
        seq = ''.join([base.upper() for base in sequence if base.upper() in 'ATGCN'])
        if not seq:
            seq = 'N'
        
        # Pad/truncate - for soft clips, center the sequence if it's shorter
        if len(seq) > self.max_seq_length:
            # If too long, take the middle part
            start = (len(seq) - self.max_seq_length) // 2
            seq = seq[start:start + self.max_seq_length]
        else:
            # If too short, pad equally on both sides
            pad_total = self.max_seq_length - len(seq)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            seq = 'N' * pad_left + seq + 'N' * pad_right
        
        # One-hot encode
        encoded = np.zeros((5, len(seq)))
        for i, base in enumerate(seq):
            if base in mapping:
                encoded[mapping[base], i] = 1
            else:
                encoded[4, i] = 1
        
        return torch.FloatTensor(encoded).to(self.device)
    
    def _encode_site_sequences(self, site_sequences: List[str]) -> torch.Tensor:
        """Encode sequences for a single site."""
        encoded_sequences = []
        
        for seq in site_sequences[:15]:  # Limit sequences per site
            if seq and seq != 'nan' and len(seq.strip()) >= 5:  # Require minimum length of 5
                encoded_seq = self._encode_sequence(seq.strip())
                encoded_sequences.append(encoded_seq)
        
        if len(encoded_sequences) == 0:
            # Return empty tensor if no valid sequences
            return torch.empty(0, 5, self.max_seq_length, device=self.device)
        
        # Stack sequences for this site
        return torch.stack(encoded_sequences)
    
    def train(self, site_data: List[Dict], validation_split: float = 0.2, 
              epochs: int = 50, batch_size: int = 16, verbose: bool = True) -> Dict:
        """Train the model."""
        print(f"Training {self.clip_type} model with {len(site_data)} sites")
        
        # Split data
        import random
        random.shuffle(site_data)
        split_idx = int(len(site_data) * (1 - validation_split))
        train_data = site_data[:split_idx]
        val_data = site_data[split_idx:]
        
        print(f"Training sites: {len(train_data)}, Validation sites: {len(val_data)}")
        
        # Check data balance
        train_pos = sum(1 for site in train_data if site['label'] == 1)
        train_neg = len(train_data) - train_pos
        val_pos = sum(1 for site in val_data if site['label'] == 1)
        val_neg = len(val_data) - val_pos
        
        print(f"Train balance: {train_pos} pos, {train_neg} neg")
        print(f"Val balance: {val_pos} pos, {val_neg} neg")
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Early stopping
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Shuffle training data
            random.shuffle(train_data)
            
            # Process sites with proper batching
            self.optimizer.zero_grad()
            batch_loss = 0.0
            batch_count = 0
            
            for i, site in enumerate(train_data):
                # Encode sequences for this site
                sequences_tensor = self._encode_site_sequences(site['sequences'])
                
                if sequences_tensor.shape[0] == 0:
                    continue  # Skip sites with no valid sequences
                
                # Forward pass for this site
                output = self.model(sequences_tensor, return_embeddings=False)
                label = torch.tensor(float(site['label']), device=self.device)
                
                # Loss for this site
                site_loss = self.criterion(output, label)
                batch_loss += site_loss
                batch_count += 1
                
                # Metrics
                prediction = torch.sigmoid(output).item()
                pred_binary = 1 if prediction > 0.5 else 0
                train_correct += (pred_binary == site['label'])
                train_total += 1
                train_loss += site_loss.item()
                
                # Update every batch_size sites
                if batch_count == batch_size or i == len(train_data) - 1:
                    # Average the batch loss
                    avg_batch_loss = batch_loss / batch_count
                    avg_batch_loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Reset batch accumulators
                    batch_loss = 0.0
                    batch_count = 0
                
                # Memory cleanup
                del sequences_tensor, output, label, site_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Training metrics
            avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
            train_accuracy = train_correct / train_total if train_total > 0 else 0.0
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            
            # Validation
            if val_data:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for site in val_data:
                        # Encode sequences for this site
                        sequences_tensor = self._encode_site_sequences(site['sequences'])
                        
                        if sequences_tensor.shape[0] == 0:
                            continue
                        
                        # Forward pass
                        output = self.model(sequences_tensor, return_embeddings=False)
                        label = torch.tensor(float(site['label']), device=self.device)
                        
                        # Loss
                        site_loss = self.criterion(output, label)
                        val_loss += site_loss.item()
                        
                        # Metrics
                        prediction = torch.sigmoid(output).item()
                        pred_binary = 1 if prediction > 0.5 else 0
                        val_correct += (pred_binary == site['label'])
                        val_total += 1
                        
                        # Memory cleanup
                        del sequences_tensor, output, label, site_loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                val_loss = val_loss / val_total if val_total > 0 else 0.0
                val_accuracy = val_correct / val_total if val_total > 0 else 0.0
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_accuracy)
                
                # Early stopping
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 5 == 0:  # Print every 5 epochs
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        
        return history
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


# Global embedder instances for 4 models
_global_site_cnn_start_tss_embedder = None
_global_site_cnn_start_tes_embedder = None
_global_site_cnn_end_tss_embedder = None
_global_site_cnn_end_tes_embedder = None

def get_site_cnn_start_tss_embedder(model_path=None) -> SiteCNNEmbedder:
    """Get global start TSS embedder."""
    global _global_site_cnn_start_tss_embedder
    
    if _global_site_cnn_start_tss_embedder is None:
        _global_site_cnn_start_tss_embedder = SiteCNNEmbedder(model_path, clip_type='start_TSS')
    return _global_site_cnn_start_tss_embedder

def get_site_cnn_start_tes_embedder(model_path=None) -> SiteCNNEmbedder:
    """Get global start TES embedder."""
    global _global_site_cnn_start_tes_embedder
    
    if _global_site_cnn_start_tes_embedder is None:
        _global_site_cnn_start_tes_embedder = SiteCNNEmbedder(model_path, clip_type='start_TES')
    return _global_site_cnn_start_tes_embedder

def get_site_cnn_end_tss_embedder(model_path=None) -> SiteCNNEmbedder:
    """Get global end TSS embedder."""
    global _global_site_cnn_end_tss_embedder
    
    if _global_site_cnn_end_tss_embedder is None:
        _global_site_cnn_end_tss_embedder = SiteCNNEmbedder(model_path, clip_type='end_TSS')
    return _global_site_cnn_end_tss_embedder

def get_site_cnn_end_tes_embedder(model_path=None) -> SiteCNNEmbedder:
    """Get global end TES embedder."""
    global _global_site_cnn_end_tes_embedder
    
    if _global_site_cnn_end_tes_embedder is None:
        _global_site_cnn_end_tes_embedder = SiteCNNEmbedder(model_path, clip_type='end_TES')
    return _global_site_cnn_end_tes_embedder

def get_separate_site_cnn_embedders(start_tss_path=None, start_tes_path=None, end_tss_path=None, end_tes_path=None):
    """Get all 4 embedders."""
    start_tss = get_site_cnn_start_tss_embedder(start_tss_path)
    start_tes = get_site_cnn_start_tes_embedder(start_tes_path)
    end_tss = get_site_cnn_end_tss_embedder(end_tss_path)
    end_tes = get_site_cnn_end_tes_embedder(end_tes_path)
    return start_tss, start_tes, end_tss, end_tes

def cleanup_site_cnn_embedder():
    """Clean up all global embedders."""
    global _global_site_cnn_start_tss_embedder, _global_site_cnn_start_tes_embedder
    global _global_site_cnn_end_tss_embedder, _global_site_cnn_end_tes_embedder
    
    for embedder_var in [_global_site_cnn_start_tss_embedder, _global_site_cnn_start_tes_embedder,
                         _global_site_cnn_end_tss_embedder, _global_site_cnn_end_tes_embedder]:
        if embedder_var is not None:
            del embedder_var
    
    _global_site_cnn_start_tss_embedder = None
    _global_site_cnn_start_tes_embedder = None
    _global_site_cnn_end_tss_embedder = None
    _global_site_cnn_end_tes_embedder = None
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()