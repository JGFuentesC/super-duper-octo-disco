#!/usr/bin/env python3
"""
Quick Demo of the Simple LSTM Bible-Quran Generative Model
Fast training and lightweight text generation.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def lstm_quick_demo():
    """Quick demonstration of the LSTM model capabilities"""
    
    print("⚡ SIMPLE LSTM BIBLE-QURAN GENERATIVE MODEL - QUICK DEMO")
    print("=" * 70)
    print()
    
    print("🚀 What this LSTM model can do:")
    print("• Train on 6000 religious texts from Bible and Quran")
    print("• Generate text continuations in Spanish")
    print("• Automatically determine optimal text length (20-100 tokens)")
    print("• Use religious and spiritual language patterns")
    print("• Train 10-20x faster than GPT-2")
    print()
    
    print("⚡ LSTM ADVANTAGES over GPT-2:")
    print("• Training time: ~5-10 minutes vs ~2-4 hours")
    print("• Model size: ~1MB vs ~500MB")
    print("• Memory usage: ~100MB vs ~2GB")
    print("• Generation speed: Real-time vs slower")
    print("• Resource requirements: CPU only vs GPU recommended")
    print()
    
    print("🎯 Example prompts that work well:")
    print("• 'Dios es amor y misericordia'")
    print("• 'La fe mueve montañas'")
    print("• 'Bendito sea el nombre del Señor'")
    print("• 'El amor de Dios es infinito'")
    print("• 'La sabiduría viene del cielo'")
    print()
    
    print("⚙️ How to use:")
    print("1. Train the LSTM model (FAST!):")
    print("   python simpleLstmModel.py --mode train")
    print()
    print("2. Generate text interactively:")
    print("   python interactiveLstmGenerator.py")
    print()
    print("3. Generate from command line:")
    print("   python simpleLstmModel.py --mode generate --prompt 'Dios es amor' --tokens 50")
    print()
    
    print("🔧 Technical details:")
    print("• Architecture: 2-layer LSTM with embeddings")
    print("• Vocabulary: Character-level (faster processing)")
    print("• Dataset: 3342 clean religious texts")
    print("• Language: Spanish religious/spiritual")
    print("• Token range: 20-100 tokens")
    print("• Memory usage: ~100MB")
    print("• Training time: 5-10 minutes on CPU")
    print()
    
    print("💡 Pro tips:")
    print("• Use religious/spiritual language for best results")
    print("• Longer prompts get shorter, focused continuations")
    print("• Shorter prompts get longer, creative continuations")
    print("• Train for 10-15 epochs for good quality")
    print("• Can retrain quickly if you want to improve")
    print()
    
    print("⚡ SPEED COMPARISON:")
    print("-" * 30)
    print("GPT-2: 2-4 hours training, 500MB model")
    print("LSTM:  5-10 minutes training, 1MB model")
    print("Result: 20-50x faster training!")
    print()
    
    print("🎉 Ready to generate religious text FAST!")
    print("Start with: python simpleLstmModel.py --mode train")

if __name__ == "__main__":
    lstm_quick_demo()
