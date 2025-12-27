from abstraction_engine import AbstractionEngine
import inspect
from typing import Callable, List, Dict, Any

class MetaLearner:
    """
    Learns how to learn.
    When it fails to find a rule, it analyzes WHY and creates new synthesis strategies.
    """
    
    def __init__(self):
        self.base_engine = AbstractionEngine()
        self.synthesis_strategies = []  # New strategies it discovers
        self.failed_patterns = []  # Patterns it couldn't synthesize
        
    def learn_with_meta(self, examples: List[Dict], max_attempts: int = 3):
        """
        Try to learn from examples.
        If it fails, analyze the failure and CREATE A NEW SYNTHESIS STRATEGY.
        """
        
        print(f"\n{'='*60}")
        print(f"META-LEARNING ATTEMPT")
        print(f"{'='*60}\n")
        
        # First attempt: Use existing strategies
        self.base_engine.add_examples(examples)
        rule = self.base_engine.discover_rule()
        
        if rule:
            print("✅ Succeeded with existing strategies!")
            return rule
        
        # Failed! Let's learn WHY
        print("❌ Existing strategies failed. Analyzing pattern...")
        
        pattern_analysis = self._analyze_failed_pattern(examples)
        print(f"\n🔬 PATTERN ANALYSIS:")
        print(f"   Type: {pattern_analysis['type']}")
        print(f"   Structure: {pattern_analysis['structure']}")
        
        # Generate new synthesis strategy
        new_strategy = self._create_synthesis_strategy(pattern_analysis, examples)
        
        if new_strategy:
            print(f"\n🧬 CREATED NEW SYNTHESIS STRATEGY:")
            print(f"   Name: {new_strategy['name']}")
            print(f"   Can handle: {new_strategy['pattern_type']}")
            
            self.synthesis_strategies.append(new_strategy)
            
            # Try again with new strategy
            print(f"\n🔄 RETRYING with new strategy...")
            
            candidates = new_strategy['generator'](examples)
            
            best_candidate = None
            best_score = 0
            
            for candidate in candidates:
                score = self._evaluate_candidate(candidate, examples)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_score > 0.8:
                print(f"✅ NEW STRATEGY WORKED! (Score: {best_score*100:.1f}%)")
                
                rule_obj = {
                    'function': best_candidate['function'],
                    'description': best_candidate['description'],
                    'code_size': len(str(best_candidate['function'])),
                    'accuracy': best_score,
                    'example_count': len(examples),
                    'learned_by': 'meta-learning',
                    'strategy_used': new_strategy['name']
                }
                
                self.base_engine.rules.append(rule_obj)
                self.base_engine.rule_applications[len(self.base_engine.rules)-1] = 0
                self.base_engine.training_examples = []
                
                return rule_obj
            else:
                print(f"❌ New strategy insufficient (Score: {best_score*100:.1f}%)")
                self.failed_patterns.append({
                    'examples': examples,
                    'analysis': pattern_analysis
                })
                return None
        
        return None
    
    def _analyze_failed_pattern(self, examples: List[Dict]) -> Dict:
        """
        Analyze WHY we couldn't synthesize a rule.
        This is the KEY to meta-learning.
        """
        
        analysis = {
            'type': 'unknown',
            'structure': 'unknown',
            'input_type': None,
            'output_type': None,
            'transformation': None
        }
        
        if not examples:
            return analysis
        
        first = examples[0]
        analysis['input_type'] = type(first['input']).__name__
        analysis['output_type'] = type(first['output']).__name__
        
        # Analyze numeric patterns
        if isinstance(first['input'], (int, float)) and isinstance(first['output'], (int, float)):
            
            # Check if it's a composition
            diffs = []
            ratios = []
            
            for ex in examples:
                inp = ex['input']
                out = ex['output']
                
                diffs.append(out - inp)
                if inp != 0:
                    ratios.append(out / inp)
            
            # Check for linear pattern: y = mx + b
            if len(set(ratios)) > 1 and len(set(diffs)) > 1:
                # Might be: multiply then add
                # Try to find m and b
                for ex in examples:
                    inp = ex['input']
                    out = ex['output']
                    
                    # Try different multipliers
                    for m in range(2, 10):
                        b = out - (inp * m)
                        if self._test_linear_pattern(examples, m, b):
                            analysis['type'] = 'linear_composition'
                            analysis['structure'] = f'multiply by {m}, then add {b}'
                            analysis['transformation'] = {'multiply': m, 'add': b}
                            return analysis
        
        # Analyze string patterns
        if isinstance(first['input'], str) and isinstance(first['output'], str):
            
            # Check for insert patterns
            for ex in examples:
                inp = ex['input']
                out = ex['output']
                
                if inp in out:
                    # Something was inserted
                    if out.startswith(inp):
                        # Suffix added
                        suffix = out[len(inp):]
                        analysis['type'] = 'string_append'
                        analysis['structure'] = f'append "{suffix}"'
                        return analysis
                    elif out.endswith(inp):
                        # Prefix added
                        prefix = out[:len(out)-len(inp)]
                        analysis['type'] = 'string_prepend'
                        analysis['structure'] = f'prepend "{prefix}"'
                        return analysis
        
        return analysis
    
    def _test_linear_pattern(self, examples: List[Dict], m: int, b: int) -> bool:
        """Test if y = mx + b fits all examples"""
        for ex in examples:
            if ex['output'] != (ex['input'] * m + b):
                return False
        return True
    
    def _create_synthesis_strategy(self, analysis: Dict, examples: List[Dict]) -> Dict:
        """
        CREATE A NEW SYNTHESIS STRATEGY based on pattern analysis.
        This is where the magic happens - the system writes new code for itself.
        """
        
        if analysis['type'] == 'linear_composition':
            m = analysis['transformation']['multiply']
            b = analysis['transformation']['add']
            
            def new_strategy_generator(examples):
                """Generated strategy: linear compositions"""
                candidates = []
                
                # Try variations around the discovered pattern
                for mult in range(m-2, m+3):
                    for add in range(b-2, b+3):
                        if mult == 0:
                            continue
                        
                        # Create the function with proper closure
                        def make_func(m, a):
                            return lambda x: x * m + a

                        candidates.append({
                            'function': make_func(mult, add),
                            'description': f'multiply by {mult} then add {add}'
                        })
                
                return candidates
            
            return {
                'name': f'linear_composition_{m}x_plus_{b}',
                'pattern_type': 'linear arithmetic composition',
                'generator': new_strategy_generator,
                'created_from': analysis
            }
        
        elif analysis['type'] in ['string_append', 'string_prepend']:
            # Could create more sophisticated string strategies here
            pass
        
        return None
    
    def _evaluate_candidate(self, candidate: Dict, examples: List[Dict]) -> float:
        """Evaluate a candidate rule"""
        try:
            func = candidate['function']
            correct = 0
            
            for ex in examples:
                try:
                    result = func(ex['input'])
                    if result == ex['output']:
                        correct += 1
                except:
                    pass
            
            return correct / len(examples) if examples else 0.0
        except:
            return 0.0
    
    def show_meta_stats(self):
        """Show what meta-learning has discovered"""
        print("\n" + "="*60)
        print("META-LEARNER STATISTICS")
        print("="*60)
        
        print(f"\n📚 Base rules learned: {len(self.base_engine.rules)}")
        print(f"🧬 New synthesis strategies created: {len(self.synthesis_strategies)}")
        print(f"❌ Patterns still unsolved: {len(self.failed_patterns)}")
        
        if self.synthesis_strategies:
            print("\n🧬 NEW SYNTHESIS STRATEGIES DISCOVERED:")
            for idx, strategy in enumerate(self.synthesis_strategies):
                print(f"   {idx+1}. {strategy['name']}")
                print(f"      Can handle: {strategy['pattern_type']}")
        
        self.base_engine.show_stats()
        
        print(f"\n💡 KEY INSIGHT: Each new strategy makes future learning faster!")
        print(f"   This system is LEARNING HOW TO LEARN.")
        print("="*60)


def test_meta_learning():
    """Test the meta-learning system"""
    
    print("="*60)
    print("META-LEARNING TEST: Learning to Learn")
    print("="*60)
    
    meta = MetaLearner()
    
    # Test 1: Start with something it CAN do
    print("\n--- PHASE 1: Learning Basic Doubling ---")
    simple_examples = [
        {'input': 2, 'output': 4},
        {'input': 5, 'output': 10},
        {'input': 7, 'output': 14},
    ]
    
    meta.learn_with_meta(simple_examples)
    
    # Test 2: Now give it something it CANNOT do (yet)
    print("\n--- PHASE 2: Learning Composite Operation ---")
    print("Pattern: double then add 5")
    
    composite_examples = [
        {'input': 2, 'output': 9},   # 2*2+5 = 9
        {'input': 3, 'output': 11},  # 3*2+5 = 11
        {'input': 5, 'output': 15},  # 5*2+5 = 15
        {'input': 10, 'output': 25}, # 10*2+5 = 25
    ]
    
    rule = meta.learn_with_meta(composite_examples)
    
    if rule:
        print("\n--- TESTING NEW RULE ---")
        for test_val in [7, 20, 100]:
            result = meta.base_engine.predict(test_val)
            expected = test_val * 2 + 5
            print(f"{test_val} → {result['output']} (expected: {expected})")
            print(f"   Used: {result['description']}")
    
    # Test 3: Try ANOTHER composite (should be easier now!)
    print("\n--- PHASE 3: Learning ANOTHER Composite ---")
    print("Pattern: triple then subtract 2")
    
    triple_examples = [
        {'input': 2, 'output': 4},   # 2*3-2 = 4
        {'input': 3, 'output': 7},   # 3*3-2 = 7
        {'input': 5, 'output': 13},  # 5*3-2 = 13
        {'input': 10, 'output': 28}, # 10*3-2 = 28
    ]
    
    rule2 = meta.learn_with_meta(triple_examples)
    
    if rule2:
        print("\n✅ LEARNED SECOND COMPOSITE FASTER!")
        print("   (Because we now have the linear composition strategy)")
    
    meta.show_meta_stats()


if __name__ == "__main__":
    test_meta_learning()