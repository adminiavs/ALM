#!/usr/bin/env python3
"""
Comprehensive diagnostic test for Recursive Intelligence system.
Tests all components, diagnoses issues, and provides complete system report.
"""

from recursive_intelligence import RecursiveIntelligence
from abstraction_engine import AbstractionEngine
from meta_learner import MetaLearner
import traceback

def test_basic_operations():
    """Test all basic arithmetic operations"""
    print("="*80)
    print("TESTING BASIC OPERATIONS")
    print("="*80)

    operations = {
        'add': [
            {'input': (2, 3), 'output': 5},
            {'input': (7, 4), 'output': 11},
            {'input': (10, 15), 'output': 25}
        ],
        'subtract': [
            {'input': (10, 3), 'output': 7},
            {'input': (15, 4), 'output': 11},
            {'input': (25, 7), 'output': 18}
        ],
        'multiply': [
            {'input': (2, 3), 'output': 6},
            {'input': (4, 5), 'output': 20},
            {'input': (6, 7), 'output': 42}
        ],
        'divide': [
            {'input': (10, 2), 'output': 5.0},
            {'input': (20, 4), 'output': 5.0},
            {'input': (30, 6), 'output': 5.0}
        ]
    }

    results = {}

    for op_name, examples in operations.items():
        print(f"\n--- Testing {op_name.upper()} operation ---")

        engine = AbstractionEngine()
        engine.add_examples(examples)

        rule = engine.discover_rule()
        results[op_name] = rule is not None

        if rule:
            print(f"✅ Successfully learned {op_name}")
            # Test on new examples
            test_cases = [
                (1, 2) if op_name in ['add', 'multiply'] else (8, 2),
                (3, 4) if op_name in ['add', 'multiply'] else (12, 3),
            ]

            for inp in test_cases:
                prediction = engine.predict(inp)
                if isinstance(prediction, dict):
                    print(f"  {inp} → {prediction['output']}")
                else:
                    print(f"  {inp} → {prediction}")
        else:
            print(f"❌ Failed to learn {op_name}")

    return results

def test_self_training_mechanism():
    """Test the self-training mechanism directly"""
    print("\n" + "="*80)
    print("TESTING SELF-TRAINING MECHANISM")
    print("="*80)

    # Create RI instance and manually test self-training
    ri = RecursiveIntelligence()

    # Add some basic knowledge first
    print("\n--- Adding basic knowledge ---")
    basic_examples = [
        {'input': (2, 3), 'output': 5},  # add
        {'input': (7, 4), 'output': 11}, # add
        {'input': (10, 3), 'output': 7}, # subtract
    ]
    ri.meta_learner.learn_with_meta(basic_examples)

    # Now test self-training with generated exercises
    print("\n--- Testing self-training ---")

    # Create a simple exercise manually
    exercise = {
        'description': 'Learn multiplication by 2',
        'examples': [
            {'input': 3, 'output': 6},
            {'input': 4, 'output': 8},
            {'input': 5, 'output': 10}
        ],
        'type': 'basic_operation',
        'target': 'multiply'
    }

    print(f"Attempting exercise: {exercise['description']}")

    try:
        rule = ri.meta_learner.learn_with_meta(exercise['examples'])
        if rule:
            print(f"✅ Self-training successful: {rule['description']}")
            return True
        else:
            print("❌ Self-training failed")
            return False
    except Exception as e:
        print(f"❌ Self-training error: {e}")
        traceback.print_exc()
        return False

def test_domain_transfer():
    """Test cross-domain transfer mechanisms"""
    print("\n" + "="*80)
    print("TESTING DOMAIN TRANSFER")
    print("="*80)

    ri = RecursiveIntelligence()

    # Learn arithmetic pattern
    print("\n--- Learning arithmetic pattern ---")
    arithmetic_examples = [
        {'input': 2, 'output': 4},
        {'input': 3, 'output': 6},
        {'input': 5, 'output': 10}
    ]
    rule1 = ri.meta_learner.learn_with_meta(arithmetic_examples)

    if rule1:
        print("✅ Learned arithmetic pattern")
    else:
        print("❌ Failed to learn arithmetic pattern")
        return False

    # Create transfer rule
    print("\n--- Creating transfer rule ---")
    transfer_rule = ri.discover("rotation is like addition in angular space")
    if transfer_rule:
        print("✅ Created transfer rule")
    else:
        print("❌ Failed to create transfer rule")
        return False

    # Test transfer capability - check if existing rule can handle transferred domain
    print("\n--- Testing transfer capability ---")
    image_examples = [
        {'input': "30_deg", 'output': "60_deg"},
        {'input': "45_deg", 'output': "90_deg"},
        {'input': "15_deg", 'output': "30_deg"}
    ]

    # Test if transfer allows existing rule to work on new domain
    success_count = 0
    for ex in image_examples:
        transformed_input = transfer_rule['transfer_function'](ex['input'])
        transformed_output = transfer_rule['transfer_function'](ex['output'])

        # Test with existing rule
        prediction = ri.meta_learner.base_engine.predict(transformed_input)
        if isinstance(prediction, dict) and prediction['output'] == transformed_output:
            success_count += 1

    transfer_success_rate = success_count / len(image_examples)
    if transfer_success_rate >= 0.8:  # 80% success rate
        print(f"✅ Domain transfer successful ({transfer_success_rate*100:.1f}% accuracy)")
        return True
    else:
        print(f"❌ Domain transfer insufficient ({transfer_success_rate*100:.1f}% accuracy)")
        return False

def test_comprehensive_ri():
    """Run comprehensive Recursive Intelligence test"""
    print("\n" + "="*80)
    print("COMPREHENSIVE RECURSIVE INTELLIGENCE TEST")
    print("="*80)

    ri = RecursiveIntelligence()

    try:
        # Test all phases
        print("\n--- PHASE 1: Basic Learning ---")
        basic_examples = [
            {'input': (2, 3), 'output': 5},
            {'input': (4, 5), 'output': 9}
        ]
        rule = ri.meta_learner.learn_with_meta(basic_examples)
        phase1_success = rule is not None

        print("\n--- PHASE 2: Abstraction ---")
        ri.abstract("arithmetic_operations")
        phase2_success = len(ri.categories) > 0

        print("\n--- PHASE 3: Knowledge Gaps ---")
        gaps = ri.analyze_knowledge_gaps()
        phase3_success = len(gaps) > 0

        print("\n--- PHASE 4: Exercise Generation ---")
        exercises = ri.generate_exercises(2)
        phase4_success = len(exercises) > 0

        print("\n--- PHASE 5: Self-Training ---")
        results = ri.self_train()
        phase5_success = any(r['success'] for r in results) if results else False

        print("\n--- PHASE 6: Language Learning ---")
        sentence_examples = [
            {'input': "Cat sits", 'output': "subject-verb"},
            {'input': "Dog runs", 'output': "subject-verb"}
        ]
        grammar_rule = ri.learn_grammar_rules(sentence_examples)
        phase6_success = grammar_rule is not None

        return {
            'phase1_basic_learning': phase1_success,
            'phase2_abstraction': phase2_success,
            'phase3_knowledge_gaps': phase3_success,
            'phase4_exercise_generation': phase4_success,
            'phase5_self_training': phase5_success,
            'phase6_language': phase6_success
        }

    except Exception as e:
        print(f"❌ Comprehensive test failed with error: {e}")
        traceback.print_exc()
        return None

def run_complete_diagnosis():
    """Run complete system diagnosis"""
    print("🔬 STARTING COMPLETE SYSTEM DIAGNOSIS")
    print("="*80)

    results = {}

    # Test basic operations
    results['basic_operations'] = test_basic_operations()

    # Test self-training mechanism
    results['self_training'] = test_self_training_mechanism()

    # Test domain transfer
    results['domain_transfer'] = test_domain_transfer()

    # Comprehensive RI test
    results['comprehensive_ri'] = test_comprehensive_ri()

    # Generate report
    generate_diagnosis_report(results)

def generate_diagnosis_report(results):
    """Generate comprehensive diagnosis report"""
    print("\n" + "="*100)
    print("🔬 COMPLETE SYSTEM DIAGNOSIS REPORT")
    print("="*100)

    # Basic Operations Report
    print("\n📊 BASIC OPERATIONS STATUS:")
    if 'basic_operations' in results:
        ops = results['basic_operations']
        for op, success in ops.items():
            status = "✅" if success else "❌"
            print(f"   {status} {op.upper()}: {'WORKING' if success else 'FAILED'}")

        working_ops = sum(1 for s in ops.values() if s)
        print(f"\n   OVERALL: {working_ops}/{len(ops)} operations working")

    # Self-Training Report
    print("\n🤖 SELF-TRAINING STATUS:")
    if 'self_training' in results:
        status = "✅ WORKING" if results['self_training'] else "❌ BROKEN"
        print(f"   Status: {status}")
        if not results['self_training']:
            print("   ISSUE: Meta-learner cannot learn from generated exercises")

    # Domain Transfer Report
    print("\n🌐 DOMAIN TRANSFER STATUS:")
    if 'domain_transfer' in results:
        status = "✅ WORKING" if results['domain_transfer'] else "❌ BROKEN"
        print(f"   Status: {status}")
        if not results['domain_transfer']:
            print("   ISSUE: Cross-domain analogies not being applied correctly")

    # Comprehensive RI Report
    print("\n🧠 RECURSIVE INTELLIGENCE PHASES:")
    if 'comprehensive_ri' in results and results['comprehensive_ri']:
        phases = results['comprehensive_ri']
        phase_names = {
            'phase1_basic_learning': 'Basic Learning',
            'phase2_abstraction': 'Hierarchical Abstraction',
            'phase3_knowledge_gaps': 'Knowledge Gap Analysis',
            'phase4_exercise_generation': 'Curriculum Generation',
            'phase5_self_training': 'Self-Training',
            'phase6_language': 'Language Understanding'
        }

        working_phases = 0
        for phase_key, success in phases.items():
            status = "✅" if success else "❌"
            name = phase_names.get(phase_key, phase_key)
            print(f"   {status} {name}")
            if success:
                working_phases += 1

        print(f"\n   OVERALL: {working_phases}/{len(phases)} phases working")

    # System Health Score
    health_score = calculate_health_score(results)
    print("\n🏥 SYSTEM HEALTH SCORE:")
    print(f"   Score: {health_score}/100")

    if health_score >= 80:
        print("   Status: 🟢 EXCELLENT")
    elif health_score >= 60:
        print("   Status: 🟡 GOOD")
    elif health_score >= 40:
        print("   Status: 🟠 FAIR")
    else:
        print("   Status: 🔴 NEEDS ATTENTION")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    generate_recommendations(results)

    print("\n" + "="*100)

def calculate_health_score(results):
    """Calculate overall system health score"""
    score = 0
    max_score = 0

    # Basic operations (40 points)
    if 'basic_operations' in results:
        ops = results['basic_operations']
        score += sum(10 for s in ops.values() if s)
        max_score += 40

    # Self-training (20 points)
    if 'self_training' in results:
        score += 20 if results['self_training'] else 0
        max_score += 20

    # Domain transfer (20 points)
    if 'domain_transfer' in results:
        score += 20 if results['domain_transfer'] else 0
        max_score += 20

    # RI phases (20 points)
    if 'comprehensive_ri' in results and results['comprehensive_ri']:
        phases = results['comprehensive_ri']
        score += sum(3.33 for s in phases.values() if s)  # ~20 points total
        max_score += 20

    return int((score / max_score) * 100) if max_score > 0 else 0

def generate_recommendations(results):
    """Generate specific recommendations based on test results"""

    issues = []

    # Check basic operations
    if 'basic_operations' in results:
        ops = results['basic_operations']
        failed_ops = [op for op, success in ops.items() if not success]
        if failed_ops:
            issues.append(f"Fix basic operations: {', '.join(failed_ops)}")

    # Check self-training
    if 'self_training' in results and not results['self_training']:
        issues.append("Debug self-training mechanism - exercises not being learned")

    # Check domain transfer
    if 'domain_transfer' in results and not results['domain_transfer']:
        issues.append("Fix domain transfer - analogies not applying correctly")

    # Check RI phases
    if 'comprehensive_ri' in results and results['comprehensive_ri']:
        phases = results['comprehensive_ri']
        failed_phases = [k for k, v in phases.items() if not v]
        if failed_phases:
            issues.append(f"Fix RI phases: {', '.join(failed_phases)}")

    if issues:
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("   • System is functioning correctly! 🎉")

if __name__ == "__main__":
    run_complete_diagnosis()
