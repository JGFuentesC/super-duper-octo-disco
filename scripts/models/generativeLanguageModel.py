#!/usr/bin/env python3
"""
Simple LSTM Generative Language Model for Bible-Quran Dataset
Fast training with a lightweight LSTM architecture.
"""

import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimpleTokenizer:
    """Simple character-level tokenizer for fast processing"""
    
    def __init__(self, texts: List[str]):
        # Create vocabulary from all texts
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Create char to idx mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample characters: {list(self.char_to_idx.keys())[:20]}")
    
    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(char, 0) for char in text]
    
    def decode(self, indices: List[int]) -> str:
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])

class BibleQuranLSTMDataset(Dataset):
    """Dataset class for LSTM training"""
    
    def __init__(self, texts: List[str], tokenizer, sequence_length: int = 100):
        self.texts = texts
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.sequences = []
        
        # Create sequences for training
        for text in texts:
            if len(text) > sequence_length:
                encoded = tokenizer.encode(text)
                for i in range(0, len(encoded) - sequence_length):
                    self.sequences.append((
                        encoded[i:i + sequence_length],
                        encoded[i + 1:i + sequence_length + 1]
                    ))
        
        print(f"Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return {
            'input': torch.tensor(input_seq, dtype=torch.long),
            'target': torch.tensor(target_seq, dtype=torch.long)
        }

class SimpleLSTMModel(nn.Module):
    """Simple LSTM model for text generation"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super(SimpleLSTMModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        output = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return output, hidden
    
    def generate(self, tokenizer, prompt: str, max_length: int = 100, temperature: float = 1.0):
        """Generate text continuation"""
        self.eval()
        
        # Encode the prompt
        prompt_encoded = tokenizer.encode(prompt)
        
        with torch.no_grad():
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, 1, self.hidden_dim).to(next(self.parameters()).device)
            c0 = torch.zeros(self.num_layers, 1, self.hidden_dim).to(next(self.parameters()).device)
            hidden = (h0, c0)
            
            # Start with the prompt
            current_input = torch.tensor([prompt_encoded[-1]], dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)
            generated = prompt_encoded.copy()
            
            for _ in range(max_length):
                # Forward pass
                output, hidden = self(current_input, hidden)
                
                # Get next token probabilities
                logits = output[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
                # Update input for next iteration
                current_input = torch.tensor([[next_token]], dtype=torch.long).to(next(self.parameters()).device)
                
                # Stop if we generate too many tokens
                if len(generated) > len(prompt_encoded) + max_length:
                    break
            
            # Decode the generated sequence
            generated_text = tokenizer.decode(generated)
            return generated_text

class SimpleLSTMTrainer:
    """Trainer class for the LSTM model"""
    
    def __init__(self, model_name: str = 'simple_lstm', sequence_length: int = 100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.sequence_length = sequence_length
        self.model = None
        self.tokenizer = None
        
        # Load existing model if available
        if os.path.exists(f'models/{model_name}'):
            print("Loading existing model...")
            self.load_model(f'models/{model_name}')
        else:
            print("Will create new model during training...")
    
    def prepare_data(self, data_path: str):
        """Prepare and preprocess the dataset"""
        print("Loading dataset...")
        
        # Load the pickle file
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
        
        print(f"Dataset shape: {df.shape}")
        
        # Use the 'limpio' column (clean text) for training
        texts = df['limpio'].dropna().tolist()
        
        # Filter out very short texts
        texts = [text for text in texts if len(text) >= 20]
        
        print(f"Number of valid texts: {len(texts)}")
        print(f"Sample text: {texts[0][:100]}...")
        
        # Create tokenizer
        self.tokenizer = SimpleTokenizer(texts)
        
        # Create dataset
        self.dataset = BibleQuranLSTMDataset(texts, self.tokenizer, self.sequence_length)
        
        return self.dataset
    
    def create_model(self):
        """Create a new LSTM model"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_data first.")
        
        self.model = SimpleLSTMModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)
        
        print(f"Created LSTM model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model
    
    def train(self, data_path: str, epochs: int = 10, batch_size: int = 32, 
              learning_rate: float = 0.001, save_path: str = 'models/simple_lstm'):
        """Train the model"""
        
        # Prepare data
        dataset = self.prepare_data(data_path)
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Total batches: {len(dataloader)}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                input_seq = batch['input'].to(self.device)
                target_seq = batch['target'].to(self.device)
                
                # Forward pass
                output, _ = self.model(input_seq)
                
                # Reshape for loss calculation
                output = output.view(-1, self.tokenizer.vocab_size)
                target = target_seq.view(-1)
                
                # Calculate loss
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Progress update
                if (batch_idx + 1) % 50 == 0:
                    print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_model(f"{save_path}_epoch_{epoch + 1}")
        
        # Save final model
        self.save_model(save_path)
        print(f"Training completed! Model saved to {save_path}")
    
    def save_model(self, save_path: str):
        """Save the model and tokenizer"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), f"{save_path}_model.pth")
        
        # Save tokenizer info as dictionary
        tokenizer_info = {
            'char_to_idx': self.tokenizer.char_to_idx,
            'idx_to_char': self.tokenizer.idx_to_char,
            'vocab_size': self.tokenizer.vocab_size
        }
        with open(f"{save_path}_tokenizer.pkl", 'wb') as f:
            pickle.dump(tokenizer_info, f)
        
        print(f"Model and tokenizer saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a saved model and tokenizer"""
        # Load tokenizer info
        with open(f"{load_path}_tokenizer.pkl", 'rb') as f:
            tokenizer_info = pickle.load(f)
        
        # Recreate tokenizer
        self.tokenizer = SimpleTokenizer([])
        self.tokenizer.char_to_idx = tokenizer_info['char_to_idx']
        self.tokenizer.idx_to_char = tokenizer_info['idx_to_char']
        self.tokenizer.vocab_size = tokenizer_info['vocab_size']
        
        # Create model
        self.create_model()
        
        # Load model weights
        self.model.load_state_dict(torch.load(f"{load_path}_model.pth", map_location=self.device))
        
        print(f"Model loaded from {load_path}")
    
    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """Generate text continuation"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        generated_text = self.model.generate(
            self.tokenizer, 
            prompt, 
            max_length=max_length, 
            temperature=temperature
        )
        
        return generated_text
    
    def predict_next_tokens(self, prompt: str, num_tokens: int = 50) -> str:
        """Predict next tokens with optimal length determination"""
        
        # Determine optimal length based on prompt complexity
        prompt_chars = len(prompt)
        
        if prompt_chars < 20:
            # Short prompt, generate more tokens
            optimal_tokens = min(100, max(30, num_tokens))
        elif prompt_chars < 50:
            # Medium prompt, balanced generation
            optimal_tokens = min(80, max(25, num_tokens))
        else:
            # Long prompt, generate fewer tokens
            optimal_tokens = min(60, max(20, num_tokens))
        
        print(f"Prompt length: {prompt_chars} characters, generating {optimal_tokens} tokens...")
        
        # Generate with moderate temperature for balanced creativity
        generated_text = self.generate_text(
            prompt=prompt,
            max_length=optimal_tokens,
            temperature=0.8
        )
        
        return generated_text

def main():
    """Main function to run training or generation"""
    parser = argparse.ArgumentParser(description='Simple LSTM Bible-Quran Generative Model')
    parser.add_argument('--mode', choices=['train', 'generate'], required=True,
                       help='Mode: train the model or generate text')
    parser.add_argument('--data_path', default='data/raw/coranBibliaCleanDataset.pkl',
                       help='Path to the dataset pickle file')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--tokens', type=int, default=50, help='Number of tokens to generate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SimpleLSTMTrainer()
    
    if args.mode == 'train':
        print("Training mode selected...")
        trainer.train(
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        print("Training completed!")
        
    elif args.mode == 'generate':
        if not args.prompt:
            print("Error: --prompt is required for generation mode")
            return
        
        print("Generation mode selected...")
        print(f"Prompt: {args.prompt}")
        
        # Check if model exists
        if not os.path.exists('models/simple_lstm_model.pth'):
            print("Error: No trained model found. Please train the model first using --mode train")
            return
        
        # Load the trained model
        trainer.load_model('models/simple_lstm')
        
        # Generate text
        generated_text = trainer.predict_next_tokens(args.prompt, args.tokens)
        
        print("\n" + "="*50)
        print("GENERATED TEXT:")
        print("="*50)
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {generated_text}")
        print("="*50)

if __name__ == "__main__":
    main()
