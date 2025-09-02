#!/usr/bin/env python3
"""
Test script for the Simple LSTM Bible-Quran Model
Quick test to verify everything is working correctly.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generativeLanguageModel import SimpleLSTMTrainer

def test_lstm_model():
    """Test the LSTM model functionality"""
    
    print("🧪 TESTING SIMPLE LSTM BIBLE-QURAN MODEL")
    print("=" * 50)
    print()
    
    # Test 1: Check if model files exist
    print("✅ Test 1: Checking model files...")
    model_files = ['models/simple_lstm_model.pth', 'models/simple_lstm_tokenizer.pkl']
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"   ✓ {file_path} exists")
        else:
            print(f"   ❌ {file_path} missing")
            return False
    
    print()
    
    # Test 2: Load the model
    print("✅ Test 2: Loading trained model...")
    try:
        trainer = SimpleLSTMTrainer()
        trainer.load_model('models/simple_lstm')
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False
    
    print()
    
    # Test 3: Generate text with different prompts
    print("✅ Test 3: Testing text generation...")
    
    test_prompts = [
        "Dios es amor",
        "La fe mueve montañas", 
        "Bendito sea el nombre del Señor",
        "El amor de Dios es infinito"
    ]
    
    for prompt in test_prompts:
        try:
            generated = trainer.predict_next_tokens(prompt, 30)
            print(f"   ✓ Prompt: '{prompt}'")
            print(f"      Generated: '{generated}'")
            print()
        except Exception as e:
            print(f"   ❌ Error generating text for '{prompt}': {e}")
            return False
    
    print()
    print("🎉 ALL TESTS PASSED! The LSTM model is working correctly.")
    print()
    print("💡 You can now use:")
    print("   • python generativeLanguageModel.py --mode generate --prompt 'Your text' --tokens 50")
    print("   • python interactiveGenerator.py")
    print("   • python quickDemo.py")
    
    return True

if __name__ == "__main__":
    success = test_lstm_model()
    if not success:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
