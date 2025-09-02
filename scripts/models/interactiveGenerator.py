#!/usr/bin/env python3
"""
Interactive Text Generator for Simple LSTM Bible-Quran Model
Fast and lightweight text generation interface.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generativeLanguageModel import SimpleLSTMTrainer

def main():
    """Interactive text generation interface for LSTM model"""
    
    print("="*60)
    print("SIMPLE LSTM BIBLE-QURAN GENERATIVE MODEL")
    print("="*60)
    print()
    
    # Check if model exists
    model_files = ['models/simple_lstm_model.pth', 'models/simple_lstm_tokenizer.pkl']
    if not all(os.path.exists(f) for f in model_files):
        print("❌ No trained LSTM model found!")
        print("Please train the model first using:")
        print("python simpleLstmModel.py --mode train")
        print()
        return
    
    # Load the model
    print("🔄 Loading LSTM model...")
    try:
        trainer = SimpleLSTMTrainer()
        trainer.load_model('models/simple_lstm')
        print("✅ LSTM model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print()
    print("💡 Enter your text prompt and the LSTM model will continue it.")
    print("💡 Type 'quit' to exit.")
    print("💡 Type 'help' for usage tips.")
    print("💡 Type 'train' to retrain the model.")
    print()
    
    while True:
        try:
            # Get user input
            prompt = input("📝 Enter your prompt: ").strip()
            
            if prompt.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif prompt.lower() == 'help':
                print_help()
                continue
            elif prompt.lower() == 'train':
                retrain_model()
                continue
            elif not prompt:
                print("⚠️  Please enter a valid prompt.")
                continue
            
            # Get number of tokens
            try:
                tokens_input = input("🔢 Number of tokens to generate (20-100, default 50): ").strip()
                if tokens_input:
                    num_tokens = int(tokens_input)
                    if num_tokens < 20 or num_tokens > 100:
                        print("⚠️  Tokens must be between 20-100. Using default 50.")
                        num_tokens = 50
                else:
                    num_tokens = 50
            except ValueError:
                print("⚠️  Invalid number. Using default 50 tokens.")
                num_tokens = 50
            
            print()
            print("🔄 Generating text with LSTM...")
            print("-" * 40)
            
            # Generate text
            generated_text = trainer.predict_next_tokens(prompt, num_tokens)
            
            print()
            print("🎯 RESULT:")
            print("=" * 40)
            print(f"📝 Original prompt: {prompt}")
            print(f"🚀 Generated continuation: {generated_text}")
            print("=" * 40)
            
            # Ask if user wants to continue
            print()
            continue_input = input("🔄 Generate another text? (y/n): ").strip().lower()
            if continue_input not in ['y', 'yes', 'sí', 'si']:
                print("👋 Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

def print_help():
    """Print help information"""
    print()
    print("📚 USAGE TIPS:")
    print("-" * 30)
    print("• Write clear, specific prompts for better results")
    print("• Use religious or spiritual language for best performance")
    print("• Try prompts like:")
    print("  - 'Dios es amor y misericordia'")
    print("  - 'La fe mueve montañas'")
    print("  - 'Bendito sea el nombre del Señor'")
    print("• Adjust token count based on your needs:")
    print("  - 20-40 tokens: Short continuations")
    print("  - 50-70 tokens: Medium continuations")
    print("  - 80-100 tokens: Long continuations")
    print()
    print("⚡ LSTM ADVANTAGES:")
    print("-" * 30)
    print("• Fast training (10-20x faster than GPT-2)")
    print("• Lightweight (~1MB vs ~500MB)")
    print("• Character-level generation")
    print("• Real-time text generation")
    print()

def retrain_model():
    """Retrain the LSTM model"""
    print("\n🔄 RETRAINING LSTM MODEL")
    print("=" * 40)
    
    try:
        # Get training parameters
        epochs = input("Number of epochs (default 10): ").strip()
        epochs = int(epochs) if epochs else 10
        
        batch_size = input("Batch size (default 32): ").strip()
        batch_size = int(batch_size) if batch_size else 32
        
        lr = input("Learning rate (default 0.001): ").strip()
        lr = float(lr) if lr else 0.001
        
        print(f"\nTraining with: {epochs} epochs, batch_size={batch_size}, lr={lr}")
        
        # Initialize trainer and train
        trainer = SimpleLSTMTrainer()
        trainer.train(
            data_path='data/raw/coranBibliaCleanDataset.pkl',
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr
        )
        
        print("✅ Retraining completed!")
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        print("Please try again.")

if __name__ == "__main__":
    main()
