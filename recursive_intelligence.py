from abstraction_engine import AbstractionEngine
from meta_learner import MetaLearner
import re
from typing import List, Dict, Any, Callable
import math

class RecursiveIntelligence:
    """
    RECURSIVE INTELLIGENCE: A system that builds hierarchical abstractions
    and generates its own curriculum for self-improvement.
    """

    def __init__(self):
        self.base_engine = AbstractionEngine()
        self.meta_learner = MetaLearner()

        # Hierarchical abstraction
        self.categories = {}  # Categories like "arithmetic_operations", "string_operations"
        self.meta_categories = {}  # Meta-categories like "transformations"
        self.abstraction_hierarchy = []  # Hierarchical relationships

        # Curriculum generation
        self.knowledge_gaps = []
        self.generated_exercises = []
        self.self_training_history = []


    # ========================================================================
    # 2. HIERARCHICAL ABSTRACTION
    # ========================================================================

    def abstract(self, category_name: str, operations: List[str] = None):
        """
        Create higher-level categories of operations
        """
        print(f"\n🏗️  CREATING ABSTRACTION: {category_name}")

        if operations is None:
            # Auto-discover operations in this category
            operations = []
            for rule in self.base_engine.rules + [r for r in self.meta_learner.base_engine.rules if r not in self.base_engine.rules]:
                if self._belongs_to_category(rule['description'], category_name):
                    operations.append(rule['description'])

        category = {
            'name': category_name,
            'operations': operations,
            'common_properties': self._find_common_properties(operations),
            'size': len(operations),
            'created_from': 'abstraction'
        }

        self.categories[category_name] = category
        print(f"✅ Created category '{category_name}' with {len(operations)} operations")
        print(f"   Common properties: {category['common_properties']}")

        return category

    def discover_meta_category(self, category_names: List[str], meta_name: str):
        """
        Create meta-categories that group categories
        """
        print(f"\n🌟 DISCOVERING META-CATEGORY: {meta_name}")

        categories = [self.categories[name] for name in category_names if name in self.categories]

        # Find what these categories have in common
        meta_properties = []
        all_operations = []
        for cat in categories:
            all_operations.extend(cat['operations'])

        # Analyze patterns across categories
        common_patterns = self._analyze_cross_category_patterns(categories)

        meta_category = {
            'name': meta_name,
            'subcategories': category_names,
            'common_patterns': common_patterns,
            'total_operations': len(all_operations),
            'abstraction_level': 'meta'
        }

        self.meta_categories[meta_name] = meta_category
        self.abstraction_hierarchy.append({
            'meta': meta_name,
            'subs': category_names
        })

        print(f"✅ Created meta-category '{meta_name}'")
        print(f"   Subcategories: {category_names}")
        print(f"   Common patterns: {common_patterns}")

        return meta_category

    # ========================================================================
    # 3. CURRICULUM GENERATION
    # ========================================================================

    def analyze_knowledge_gaps(self):
        """
        Analyze what the system doesn't know yet
        """
        print(f"\n🔍 ANALYZING KNOWLEDGE GAPS...")

        gaps = []

        # Check for missing basic operations
        basic_ops = ['add', 'subtract', 'multiply', 'divide']
        learned_ops = [rule['description'] for rule in self.base_engine.rules]

        for op in basic_ops:
            found = any(op in desc for desc in learned_ops)
            if not found:
                gaps.append(f"Missing basic operation: {op}")

        # Check for missing compositions
        if not any('then' in rule['description'] for rule in self.base_engine.rules):
            gaps.append("No composition rules learned yet")


        # Check for abstractions
        if not self.categories:
            gaps.append("No abstract categories formed")

        self.knowledge_gaps = gaps

        print(f"📊 Found {len(gaps)} knowledge gaps:")
        for gap in gaps:
            print(f"   • {gap}")

        return gaps

    def generate_exercises(self, num_exercises: int = 5):
        """
        Generate exercises to fill knowledge gaps
        """
        print(f"\n📚 GENERATING EXERCISES...")

        exercises = []

        for gap in self.knowledge_gaps:
            if "basic operation" in gap:
                op = gap.split(": ")[1]
                exercises.extend(self._generate_basic_op_exercises(op, 2))

            elif "composition rules" in gap:
                exercises.extend(self._generate_composition_exercises(2))

            elif "abstract categories" in gap:
                exercises.extend(self._generate_abstraction_exercises(1))

        self.generated_exercises = exercises[:num_exercises]

        print(f"✅ Generated {len(self.generated_exercises)} exercises:")
        for i, ex in enumerate(self.generated_exercises):
            print(f"   {i+1}. {ex['description']}")

        return self.generated_exercises

    def self_train(self):
        """
        Use generated exercises to train itself
        """
        print(f"\n🏋️  SELF-TRAINING...")

        results = []
        for exercise in self.generated_exercises:
            print(f"\n📝 Attempting: {exercise['description']}")

            # Try to learn from the exercise examples
            rule = self.meta_learner.learn_with_meta(exercise['examples'])

            success = rule is not None
            results.append({
                'exercise': exercise['description'],
                'success': success,
                'rule_learned': rule['description'] if rule else None
            })

            if success:
                print(f"✅ Learned: {rule['description']}")
            else:
                print(f"❌ Failed to learn")

        self.self_training_history.extend(results)

        success_rate = sum(1 for r in results if r['success']) / len(results)
        print(f"\n📈 Self-training complete: {success_rate*100:.1f}% success rate")

        return results


    # ========================================================================
    # HELPER METHODS
    # ========================================================================


    def _belongs_to_category(self, operation_desc: str, category: str):
        """Check if operation belongs to category"""
        if category == "arithmetic_operations":
            return any(word in operation_desc for word in ['add', 'multiply', 'subtract', 'divide'])
        elif category == "string_operations":
            return any(word in operation_desc for word in ['append', 'prepend', 'uppercase'])
        return False

    def _find_common_properties(self, operations: List[str]):
        """Find common properties across operations"""
        properties = []
        if all('multiply' in op or 'add' in op for op in operations):
            properties.append("mathematical_transformation")
        if all(len(op.split()) > 1 for op in operations):
            properties.append("composed_operation")
        return properties

    def _analyze_cross_category_patterns(self, categories: List[Dict]):
        """Find patterns across categories"""
        patterns = []
        all_props = []
        for cat in categories:
            all_props.extend(cat.get('common_properties', []))

        if 'mathematical_transformation' in all_props:
            patterns.append("transformation_operations")
        if 'composed_operation' in all_props:
            patterns.append("complex_operations")

        return patterns

    def _generate_basic_op_exercises(self, operation: str, count: int):
        """Generate exercises for basic operations"""
        exercises = []
        for i in range(count):
            if operation == 'add':
                examples = [
                    {'input': (i+1, i+2), 'output': (i+1) + (i+2)},
                    {'input': (i+3, i+4), 'output': (i+3) + (i+4)},
                    {'input': (i+5, i+6), 'output': (i+5) + (i+6)}
                ]
            elif operation == 'multiply':
                examples = [
                    {'input': (i+1, i+2), 'output': (i+1) * (i+2)},
                    {'input': (i+2, i+3), 'output': (i+2) * (i+3)},
                    {'input': (i+3, i+4), 'output': (i+3) * (i+4)}
                ]
            else:
                continue

            exercises.append({
                'description': f'Learn {operation} operation (variant {i+1})',
                'examples': examples,
                'type': 'basic_operation',
                'target': operation
            })

        return exercises

    def _generate_composition_exercises(self, count: int):
        """Generate composition exercises"""
        exercises = []
        for i in range(count):
            # Generate "multiply by X then add Y" patterns
            x, y = i+2, i+5
            examples = []
            for val in [1, 2, 3, 4]:
                examples.append({
                    'input': val,
                    'output': val * x + y
                })

            exercises.append({
                'description': f'Learn composition: multiply by {x} then add {y}',
                'examples': examples,
                'type': 'composition',
                'pattern': f'{x}x+{y}'
            })

        return exercises


    def _generate_abstraction_exercises(self, count: int):
        """Generate abstraction exercises"""
        return [{
            'description': 'Create abstraction category from learned operations',
            'examples': [],  # Meta-level task
            'type': 'abstraction',
            'target': 'operation_categories'
        }]


    def show_full_status(self):
        """Show complete system status"""
        print(f"\n{'='*70}")
        print(f"RECURSIVE INTELLIGENCE - FULL STATUS")
        print(f"{'='*70}")

        print(f"\n📚 RULES LEARNED: {len(self.base_engine.rules + self.meta_learner.base_engine.rules)}")

        print(f"\n🏗️  CATEGORIES: {list(self.categories.keys())}")
        for name, cat in self.categories.items():
            print(f"   {name}: {cat['size']} operations")

        print(f"\n🌟 META-CATEGORIES: {list(self.meta_categories.keys())}")
        for name, meta in self.meta_categories.items():
            print(f"   {name}: {len(meta['subcategories'])} subcategories")

        print(f"\n📚 KNOWLEDGE GAPS: {len(self.knowledge_gaps)}")
        print(f"📝 GENERATED EXERCISES: {len(self.generated_exercises)}")
        print(f"🏋️  SELF-TRAINING SESSIONS: {len(self.self_training_history)}")

        total_size = sum(rule['code_size'] for rule in self.base_engine.rules)
        print(f"\n📊 MODEL SIZE: {total_size} bytes")
        print(f"   ({100_000_000_000 // max(total_size, 1):,}x smaller than LLM)")

        print(f"{'='*70}\n")


def demo_recursive_intelligence():
    """
    DEMO: Show advanced capabilities in action
    """
    print(f"{'='*80}")
    print(f"RECURSIVE INTELLIGENCE DEMO - THE NEXT LEVEL")
    print(f"{'='*80}\n")

    ri = RecursiveIntelligence()

    # ========================================================================
    # 1. HIERARCHICAL ABSTRACTION
    # ========================================================================
    print(f"{'='*60}")
    print(f"PHASE 1: HIERARCHICAL ABSTRACTION")
    print(f"{'='*60}\n")

    # Learn some basic operations first
    print("🎯 Learning basic operations...")
    arithmetic_examples = [
        {'input': (2, 3), 'output': 5},  # addition
        {'input': (4, 2), 'output': 2},  # subtraction
    ]
    ri.meta_learner.learn_with_meta(arithmetic_examples)

    # Create operation categories
    ri.abstract("arithmetic_operations")

    # Learn a string operation to have something to abstract
    string_examples = [
        {'input': 'cat', 'output': 'cats'},
        {'input': 'dog', 'output': 'dogs'}
    ]
    ri.meta_learner.learn_with_meta(string_examples)
    ri.abstract("string_operations")

    # Create meta-category
    ri.discover_meta_category(["arithmetic_operations", "string_operations"], "transformations")

    # ========================================================================
    # 2. CURRICULUM GENERATION
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: CURRICULUM GENERATION")
    print(f"{'='*60}\n")

    # Analyze knowledge gaps
    ri.analyze_knowledge_gaps()

    # Generate exercises
    ri.generate_exercises(3)

    # Self-train
    ri.self_train()

    # ========================================================================
    # FINAL STATUS
    # ========================================================================
    ri.show_full_status()

    print(f"{'='*80}")
    print(f"🎉 RECURSIVE INTELLIGENCE ACHIEVEMENTS:")
    print(f"   ✅ Hierarchical abstraction: {len(ri.categories)} categories, {len(ri.meta_categories)} meta-categories")
    print(f"   ✅ Curriculum generation: {len(ri.generated_exercises)} exercises generated")
    print(f"   ✅ Self-improvement: {len([r for r in ri.self_training_history if r['success']])} successful self-training sessions")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    demo_recursive_intelligence()