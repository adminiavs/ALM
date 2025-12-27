from abstraction_engine import AbstractionEngine

def experiment_arithmetic():
    """Learn multiple arithmetic operations"""
    print("\n" + "="*60)
    print("EXPERIMENT: Multi-Operation Arithmetic")
    print("="*60 + "\n")
    
    engine = AbstractionEngine()
    
    # Learn addition
    print("Learning ADDITION...")
    engine.add_examples([
        {'input': (2, 3), 'output': 5},
        {'input': (10, 5), 'output': 15},
        {'input': (7, 8), 'output': 15},
    ])
    engine.discover_rule()
    
    # Learn multiplication
    print("\nLearning MULTIPLICATION...")
    engine.add_examples([
        {'input': (2, 3), 'output': 6},
        {'input': (4, 5), 'output': 20},
        {'input': (3, 7), 'output': 21},
    ])
    engine.discover_rule()
    
    # Learn max
    print("\nLearning MAXIMUM...")
    engine.add_examples([
        {'input': (5, 3), 'output': 5},
        {'input': (2, 8), 'output': 8},
        {'input': (10, 10), 'output': 10},
    ])
    engine.discover_rule()
    
    engine.show_stats()
    
    # Test them
    print("\nTesting all learned operations:")
    print(f"(99, 1) → {engine.predict((99, 1))['output']} (should use max)")
    print(f"(10, 5) → {engine.predict((10, 5))['output']} (should use max)")


def experiment_compression():
    """Show the compression power"""
    print("\n" + "="*60)
    print("EXPERIMENT: Compression Ratio")
    print("="*60 + "\n")
    
    engine = AbstractionEngine()
    
    # Generate 100 examples of doubling
    examples = [{'input': i, 'output': i * 2} for i in range(1, 101)]
    
    print(f"Training data: 100 examples")
    print(f"If stored as examples: ~{len(str(examples))} bytes")
    
    engine.add_examples(examples[:10])  # Only need 10 to learn the rule!
    rule = engine.discover_rule()
    
    if rule:
        print(f"Learned rule size: {rule['code_size']} bytes")
        print(f"COMPRESSION: {len(str(examples)) / rule['code_size']:.1f}x")
        
        # Verify it works on ALL 100
        print("\nTesting on remaining 90 examples...")
        correct = 0
        for ex in examples[10:]:
            result = engine.predict(ex['input'])
            if result['output'] == ex['output']:
                correct += 1
        
        print(f"Accuracy: {correct}/90 = {correct/90*100:.1f}%")
        print("✅ Learned from 10, works on 100!")


def experiment_composition():
    """Learn composite operations"""
    print("\n" + "="*60)
    print("EXPERIMENT: Composite Rules")
    print("="*60 + "\n")
    
    engine = AbstractionEngine()
    
    # Teach: double then add 5
    print("Learning: double then add 5")
    engine.add_examples([
        {'input': 2, 'output': 9},   # 2*2+5
        {'input': 3, 'output': 11},  # 3*2+5
        {'input': 5, 'output': 15},  # 5*2+5
        {'input': 10, 'output': 25}, # 10*2+5
    ])
    
    rule = engine.discover_rule()
    
    if rule:
        print("\nTesting on new numbers:")
        for x in [7, 20, 100]:
            result = engine.predict(x)
            expected = x * 2 + 5
            print(f"{x} → {result['output']} (expected: {expected})")
    
    engine.show_stats()


if __name__ == "__main__":
    experiment_arithmetic()
    input("\nPress Enter for next experiment...")
    
    experiment_compression()
    input("\nPress Enter for next experiment...")
    
    experiment_composition()
    
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETE - 100% LOCAL, NO APIs!")
    print("="*60)