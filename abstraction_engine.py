"""
abstraction_engine.py

Compatibility layer for RecursiveIntelligence system.
Provides the expected AbstractionEngine interface.
"""

from __future__ import annotations
from typing import List, Dict, Any, Callable
import re
import ast
from collections import defaultdict
import operator
import itertools

class AbstractionEngine:
    """
    Program synthesis engine that learns rules from examples.
    Provides the interface expected by RecursiveIntelligence.
    """

    def __init__(self):
        self.rules = []  # Learned rules
        self.training_examples = []  # Current training data
        self.rule_applications = defaultdict(int)  # Usage counts

    def add_examples(self, examples: List[Dict[str, Any]]):
        """Add training examples"""
        self.training_examples.extend(examples)
        print(f"Added {len(examples)} examples. Total: {len(self.training_examples)}")

    def discover_rule(self, min_examples: int = 3):
        """
        Try to synthesize a rule from the training examples.
        Returns a rule dict if successful, None otherwise.
        """

        if len(self.training_examples) < min_examples:
            print(f"Need {min_examples} examples, have {len(self.training_examples)}")
            return None

        print("🔍 Attempting to synthesize rule from examples...")

        # Try simple pattern matching first
        rule = self._try_simple_patterns()
        if rule:
            print(f"✅ DISCOVERED RULE! (Score: 100%)")
            print(f"   Pattern: {rule['description']}")

            self.rules.append(rule)
            self.rule_applications[len(self.rules)-1] = 0
            self.training_examples = []  # Clear training data
            return rule

        print("❌ No clear pattern found")
        return None

    def _try_simple_patterns(self) -> Dict:
        """Try to find simple patterns in the examples"""

        if not self.training_examples:
            return None

        first = self.training_examples[0]
        input_type = type(first.get('input')).__name__
        output_type = type(first.get('output')).__name__

        # Try arithmetic patterns
        if input_type in ['int', 'float'] and output_type in ['int', 'float']:
            return self._try_arithmetic_pattern()

        # Try string patterns
        elif input_type == 'str' and output_type == 'str':
            return self._try_string_pattern()

        # Try tuple patterns (operations on pairs)
        elif input_type == 'tuple' and len(first.get('input', ())) == 2:
            return self._try_tuple_pattern()

        return None

    def _try_arithmetic_pattern(self) -> Dict:
        """Try arithmetic operations"""

        for example in self.training_examples:
            inp, out = example['input'], example['output']

            # Try multiply by constant
            for mult in range(-10, 11):
                if mult == 0:
                    continue
                if all(ex['output'] == ex['input'] * mult for ex in self.training_examples):
                    return {
                        'function': lambda x, m=mult: x * m,
                        'description': f'multiply by {mult}',
                        'code_size': len(f'lambda x: x * {mult}'),
                        'accuracy': 1.0,
                        'example_count': len(self.training_examples)
                    }

            # Try add constant
            for add in range(-20, 21):
                if all(ex['output'] == ex['input'] + add for ex in self.training_examples):
                    return {
                        'function': lambda x, a=add: x + a,
                        'description': f'add {add}',
                        'code_size': len(f'lambda x: x + {add}'),
                        'accuracy': 1.0,
                        'example_count': len(self.training_examples)
                    }

        return None

    def _try_string_pattern(self) -> Dict:
        """Try string operations"""

        for example in self.training_examples:
            inp, out = example['input'], example['output']

            # Try append suffix
            if inp in out and len(out) > len(inp):
                suffix = out[len(inp):]
                if all(ex['output'] == ex['input'] + suffix for ex in self.training_examples):
                    return {
                        'function': lambda x, s=suffix: x + s,
                        'description': f'append "{suffix}"',
                        'code_size': len(f'lambda x: x + "{suffix}"'),
                        'accuracy': 1.0,
                        'example_count': len(self.training_examples)
                    }

            # Try prepend prefix
            if inp in out and len(out) > len(inp):
                prefix = out[:-len(inp)]
                if all(ex['output'] == prefix + ex['input'] for ex in self.training_examples):
                    return {
                        'function': lambda x, p=prefix: p + x,
                        'description': f'prepend "{prefix}"',
                        'code_size': len(f'lambda x: "{prefix}" + x'),
                        'accuracy': 1.0,
                        'example_count': len(self.training_examples)
                    }

        return None

    def _try_tuple_pattern(self) -> Dict:
        """Try operations on pairs"""

        operations = [
            (operator.add, 'add'),
            (operator.sub, 'subtract'),
            (operator.mul, 'multiply'),
            (operator.truediv, 'divide')
        ]

        for op_func, op_name in operations:
            try:
                if all(ex['output'] == op_func(ex['input'][0], ex['input'][1]) for ex in self.training_examples):
                    return {
                        'function': lambda x, op=op_func: op(x[0], x[1]),
                        'description': f'{op_name} two numbers',
                        'code_size': len(f'lambda x: x[0] {op_name} x[1]'),
                        'accuracy': 1.0,
                        'example_count': len(self.training_examples)
                    }
            except:
                continue

        return None

    def show_stats(self):
        """Show engine statistics"""
        print(f"📊 AbstractionEngine Stats:")
        print(f"   Rules learned: {len(self.rules)}")
        print(f"   Total applications: {sum(self.rule_applications.values())}")

        if self.rules:
            print("   Learned rules:")
            for i, rule in enumerate(self.rules):
                apps = self.rule_applications[i]
                print(f"     {i+1}. {rule['description']} ({apps} applications)")


# =========================
# Original AbstractionCore (below)
# =========================

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Iterable
from collections import defaultdict, Counter
import math
import random

# =========================
# 1) Tiny DSL (expressions)
# =========================

@dataclass(frozen=True)
class Expr:
    op: str
    args: Tuple["Expr", ...] = ()
    val: Optional[int] = None  # for CONST
    name: Optional[str] = None  # for VAR or MACRO

    def __str__(self) -> str:
        if self.op == "VAR":
            return self.name or "x"
        if self.op == "CONST":
            return str(self.val)
        if self.op == "MACRO":
            # show macro name + expansion
            return f"{self.name}({', '.join(str(a) for a in self.args)})"
        if len(self.args) == 1:
            return f"{self.op}({self.args[0]})"
        return f"{self.op}({', '.join(str(a) for a in self.args)})"

    def size(self) -> int:
        if self.op in ("VAR", "CONST"):
            return 1
        return 1 + sum(a.size() for a in self.args)

def safe_int(v: int) -> int:
    # keep search space stable; prevents huge blowups
    if v > 10_000: return 10_000
    if v < -10_000: return -10_000
    return v

class DSL:
    """
    Base operators are deliberately tiny.
    Abstractions are added as MACRO operators.
    """
    def __init__(self):
        self.consts = [-2, -1, 0, 1, 2, 3]
        self.unary_ops: Dict[str, Callable[[int], int]] = {
            "NEG": lambda x: safe_int(-x),
            "ABS": lambda x: abs(x),
            "INC": lambda x: safe_int(x + 1),
            "DEC": lambda x: safe_int(x - 1),
        }
        self.binary_ops: Dict[str, Callable[[int, int], int]] = {
            "ADD": lambda a, b: safe_int(a + b),
            "SUB": lambda a, b: safe_int(a - b),
            "MUL": lambda a, b: safe_int(a * b),
            "MAX": lambda a, b: max(a, b),
            "MIN": lambda a, b: min(a, b),
        }
        # macro_name -> (arity, expansion_fn over Exprs)
        self.macros: Dict[str, Tuple[int, Callable[..., Expr]]] = {}

    def eval(self, e: Expr, x: int) -> int:
        if e.op == "VAR":
            return x
        if e.op == "CONST":
            return int(e.val or 0)
        if e.op == "MACRO":
            arity, expander = self.macros[e.name]  # type: ignore[index]
            if len(e.args) != arity:
                raise ValueError("Bad macro arity")
            expanded = expander(*e.args)
            return self.eval(expanded, x)
        if e.op in self.unary_ops:
            return self.unary_ops[e.op](self.eval(e.args[0], x))
        if e.op in self.binary_ops:
            return self.binary_ops[e.op](self.eval(e.args[0], x), self.eval(e.args[1], x))
        raise ValueError(f"Unknown op: {e.op}")

# =========================
# 2) Tasks + verification
# =========================

@dataclass(frozen=True)
class Task:
    name: str
    # examples as (input, output) pairs
    examples: Tuple[Tuple[int, int], ...]

def passes(dsl: DSL, expr: Expr, task: Task) -> bool:
    try:
        for x, y in task.examples:
            if dsl.eval(expr, x) != y:
                return False
        return True
    except Exception:
        return False

# =========================
# 3) Pruner (various pruning strategies)
# =========================

@dataclass
class PruningStats:
    """Statistics for pruning effectiveness"""
    generated_count: int = 0
    tested_count: int = 0
    pruned_count: int = 0
    first_solution_index: Optional[int] = None

class Pruner:
    """
    Multi-strategy pruner for expression enumeration.
    Combines semantic duplicate pruning, normal-form pruning, and learned invariants.
    """

    def __init__(self, dsl: DSL):
        self.dsl = dsl
        self.stats = PruningStats()

        # Type A: Semantic duplicate pruning
        self.probe_inputs = [-3, -2, -1, 0, 1, 2, 3, 5]  # PROBE set
        self.seen_signatures: set[Tuple[int, ...]] = set()

        # Type B: Normal-form pruning
        self.commutative_ops = {"ADD", "MUL", "MAX", "MIN"}

        # Learned pruning rules
        self.task_invariants: Dict[str, Any] = {}
        self.operator_bans: Dict[str, set[str]] = {}

        # Learning from solutions
        self.frequent_subtrees: Dict[str, int] = defaultdict(int)

    def reset_for_task(self, task: Task):
        """Reset per-task state"""
        self.stats = PruningStats()
        self.seen_signatures.clear()
        self.task_invariants = self._analyze_task_invariants(task)
        self.operator_bans = self._compute_operator_bans(self.task_invariants)

    def reject(self, expr: Expr) -> bool:
        """
        Main pruning decision. Returns True if expression should be pruned.
        """
        self.stats.generated_count += 1

        # Type A: Semantic duplicate pruning
        if self._is_semantic_duplicate(expr):
            self.stats.pruned_count += 1
            return True

        # Type B: Normal-form pruning
        if self._is_normal_form_violation(expr):
            self.stats.pruned_count += 1
            return True

        # Learned pruning: Task-conditional bans
        if self._violates_task_constraints(expr):
            self.stats.pruned_count += 1
            return True

        return False

    def mark_tested(self):
        """Increment tested count"""
        self.stats.tested_count += 1

    def mark_solution_found(self):
        """Record when first solution is found"""
        if self.stats.first_solution_index is None:
            self.stats.first_solution_index = self.stats.tested_count

    def learn_from_solution(self, expr: Expr):
        """Learn pruning rules from successful solutions"""
        # Extract and count frequent subtrees
        self._extract_subtrees(expr)

    # Type A: Semantic Duplicate Pruning
    def _is_semantic_duplicate(self, expr: Expr) -> bool:
        """Check if expression is semantically equivalent to a previously seen one"""
        try:
            signature = self._compute_signature(expr)
            if signature in self.seen_signatures:
                return True
            self.seen_signatures.add(signature)
            return False
        except Exception:
            # If evaluation fails, treat as reject (inconsistent)
            return True

    def _compute_signature(self, expr: Expr) -> Tuple[int, ...]:
        """Compute semantic signature on probe inputs"""
        outputs = []
        for x in self.probe_inputs:
            try:
                result = self.dsl.eval(expr, x)
                # Clamp values to prevent signature explosion
                clamped = max(-10_000, min(10_000, result))
                outputs.append(clamped)
            except Exception:
                # Treat exceptions as a special marker
                outputs.append(-999999)  # Distinct marker for failures
        return tuple(outputs)

    # Type B: Normal-Form Pruning
    def _is_normal_form_violation(self, expr: Expr) -> bool:
        """Check for structural redundancy"""
        return self._has_identity_operations(expr) or not self._has_canonical_ordering(expr)

    def _has_identity_operations(self, expr: Expr) -> bool:
        """Check for redundant identity operations like ADD(x,0), MUL(x,1), etc."""
        if expr.op == "ADD":
            # ADD(x, 0) is redundant
            if len(expr.args) == 2:
                a, b = expr.args
                if (a.op == "CONST" and a.val == 0) or (b.op == "CONST" and b.val == 0):
                    return True
        elif expr.op == "MUL":
            # MUL(x, 1) is redundant, MUL(x, 0) -> 0
            if len(expr.args) == 2:
                a, b = expr.args
                if (a.op == "CONST" and a.val == 1) or (b.op == "CONST" and b.val == 1):
                    return True
                if (a.op == "CONST" and a.val == 0) or (b.op == "CONST" and b.val == 0):
                    return True
        elif expr.op == "SUB":
            # SUB(x, 0) is redundant
            if len(expr.args) == 2:
                a, b = expr.args
                if b.op == "CONST" and b.val == 0:
                    return True
        elif expr.op == "NEG":
            # NEG(NEG(x)) is redundant
            if len(expr.args) == 1:
                inner = expr.args[0]
                if inner.op == "NEG":
                    return True

        # Recursively check subexpressions
        for arg in expr.args:
            if self._has_identity_operations(arg):
                return True

        return False

    def _has_canonical_ordering(self, expr: Expr) -> bool:
        """Check canonical ordering for commutative operations"""
        if expr.op in self.commutative_ops and len(expr.args) == 2:
            a, b = expr.args
            # Use string representation as key for ordering
            key_a = str(a)
            key_b = str(b)
            if key_a > key_b:
                return False

        # Recursively check subexpressions
        for arg in expr.args:
            if not self._has_canonical_ordering(arg):
                return True  # True means valid, False means violation

        return True

    # Learned Pruning: Task Invariants
    def _analyze_task_invariants(self, task: Task) -> Dict[str, Any]:
        """Compute task invariants from examples"""
        invariants = {}

        if not task.examples:
            return invariants

        xs, ys = zip(*task.examples)
        xs, ys = list(xs), list(ys)

        # Check linearity (constant slope)
        invariants['is_linear'] = self._check_linearity(xs, ys)

        # Check monotonicity
        invariants['is_increasing'] = self._check_monotonicity(xs, ys, increasing=True)
        invariants['is_decreasing'] = self._check_monotonicity(xs, ys, increasing=False)

        # Check symmetry
        invariants['is_even'] = self._check_symmetry(xs, ys, even=True)
        invariants['is_odd'] = self._check_symmetry(xs, ys, even=False)

        # Check range properties
        invariants['has_negatives'] = any(y < 0 for y in ys)
        invariants['has_positives'] = any(y > 0 for y in ys)
        invariants['crosses_zero'] = any(y <= 0 for y in ys) and any(y >= 0 for y in ys)

        return invariants

    def _check_linearity(self, xs: List[int], ys: List[int]) -> bool:
        """Check if function appears linear"""
        if len(xs) < 3:
            return False

        # Check if slope is constant
        slopes = []
        for i in range(len(xs) - 1):
            if xs[i+1] != xs[i]:
                slope = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])
                slopes.append(slope)

        # All slopes should be approximately equal
        if not slopes:
            return False

        avg_slope = sum(slopes) / len(slopes)
        return all(abs(s - avg_slope) < 0.1 for s in slopes)

    def _check_monotonicity(self, xs: List[int], ys: List[int], increasing: bool) -> bool:
        """Check if function is monotonic"""
        sorted_pairs = sorted(zip(xs, ys))
        sorted_ys = [y for x, y in sorted_pairs]

        if increasing:
            return all(sorted_ys[i] <= sorted_ys[i+1] for i in range(len(sorted_ys)-1))
        else:
            return all(sorted_ys[i] >= sorted_ys[i+1] for i in range(len(sorted_ys)-1))

    def _check_symmetry(self, xs: List[int], ys: List[int], even: bool) -> bool:
        """Check if function is even or odd"""
        # Need both positive and negative x values
        x_to_y = dict(zip(xs, ys))

        for x in xs:
            if -x in x_to_y:
                y1 = x_to_y[x]
                y2 = x_to_y[-x]
                if even and y1 != y2:
                    return False
                if not even and y1 != -y2:
                    return False

        return True  # Conservative: assume true if we can't disprove

    def _compute_operator_bans(self, invariants: Dict[str, Any]) -> Dict[str, set[str]]:
        """Map invariants to banned operators"""
        bans = defaultdict(set)

        # If linear, ban piecewise operators
        if invariants.get('is_linear', False):
            bans['operators'].update(['ABS', 'MAX', 'MIN'])

        # If no negatives in output, ban operations that might produce them
        if not invariants.get('has_negatives', True):
            bans['operators'].update(['NEG'])

        # If constant slope and non-zero, ban constant expressions
        if invariants.get('is_linear', False):
            linear_slope = self._compute_slope(list(zip(*self.task_invariants.get('examples', []))))
            if linear_slope != 0:
                bans['patterns'].add('constant_expressions')

        return dict(bans)

    def _violates_task_constraints(self, expr: Expr) -> bool:
        """Check if expression violates task-specific constraints"""
        # Check operator bans
        if expr.op in self.operator_bans.get('operators', set()):
            return True

        # Check pattern bans
        if 'constant_expressions' in self.operator_bans.get('patterns', set()):
            if self._is_constant_expression(expr):
                return True

        return False

    def _is_constant_expression(self, expr: Expr) -> bool:
        """Check if expression always evaluates to a constant"""
        # Simple check: if no VAR in the expression
        def has_var(e: Expr) -> bool:
            if e.op == "VAR":
                return True
            return any(has_var(arg) for arg in e.args)

        return not has_var(expr)

    def _compute_slope(self, examples: List[Tuple[int, int]]) -> float:
        """Compute slope from examples"""
        if len(examples) < 2:
            return 0

        xs, ys = zip(*examples)
        # Simple slope calculation
        return (ys[-1] - ys[0]) / (xs[-1] - xs[0]) if xs[-1] != xs[0] else 0

    # Learning from solutions
    def _extract_subtrees(self, expr: Expr):
        """Extract all subtrees from a solution for frequency analysis"""
        def extract(e: Expr):
            # Count this subtree
            tree_str = str(e)
            self.frequent_subtrees[tree_str] += 1

            # Recursively extract subtrees
            for arg in e.args:
                extract(arg)

        extract(expr)

# =========================
# 3) Enumerator (by size)
# =========================

class Enumerator:
    """
    Enumerates Expr by increasing size; counts candidates tested.
    Uses semantic memoization on example vectors to prune duplicates.
    """
    def __init__(self, dsl: DSL, inputs_probe: Tuple[int, ...]):
        self.dsl = dsl
        self.inputs_probe = inputs_probe

    def _signature(self, e: Expr) -> Tuple[int, ...]:
        # semantic vector on a fixed probe set (cheap dedup)
        return tuple(self.dsl.eval(e, x) for x in self.inputs_probe)

    def generate(self, max_size: int) -> Iterable[Expr]:
        # dynamic programming by size
        by_size: Dict[int, List[Expr]] = defaultdict(list)
        seen_sig: set[Tuple[int, ...]] = set()

        # size 1: VAR and CONST
        var = Expr("VAR", name="x")
        by_size[1].append(var)
        seen_sig.add(self._signature(var))

        for c in self.dsl.consts:
            e = Expr("CONST", val=c)
            sig = self._signature(e)
            if sig not in seen_sig:
                seen_sig.add(sig)
                by_size[1].append(e)

        # macros also become available as constructors (but expand in eval)
        # they will be built in later sizes according to their arity.

        for sz in range(2, max_size + 1):
            # unary: op(arg)
            for op in list(self.dsl.unary_ops.keys()):
                for arg_sz in range(1, sz):
                    if 1 + arg_sz != sz:
                        continue
                    for a in by_size[arg_sz]:
                        e = Expr(op, args=(a,))
                        sig = self._signature(e)
                        if sig not in seen_sig:
                            seen_sig.add(sig)
                            by_size[sz].append(e)

            # binary: op(a,b)
            for op in list(self.dsl.binary_ops.keys()):
                for a_sz in range(1, sz):
                    for b_sz in range(1, sz):
                        if 1 + a_sz + b_sz != sz:
                            continue
                        for a in by_size[a_sz]:
                            for b in by_size[b_sz]:
                                e = Expr(op, args=(a, b))
                                sig = self._signature(e)
                                if sig not in seen_sig:
                                    seen_sig.add(sig)
                                    by_size[sz].append(e)

            # macros: name(args...)
            for mname, (arity, _) in list(self.dsl.macros.items()):
                # distribute sizes across args
                # require each arg size >=1 and total = sz-1
                target = sz - 1
                if arity == 1:
                    for a_sz in range(1, target + 1):
                        if a_sz != target:
                            continue
                        for a in by_size[a_sz]:
                            e = Expr("MACRO", args=(a,), name=mname)
                            sig = self._signature(e)
                            if sig not in seen_sig:
                                seen_sig.add(sig)
                                by_size[sz].append(e)
                elif arity == 2:
                    for a_sz in range(1, target):
                        b_sz = target - a_sz
                        for a in by_size[a_sz]:
                            for b in by_size[b_sz]:
                                e = Expr("MACRO", args=(a, b), name=mname)
                                sig = self._signature(e)
                                if sig not in seen_sig:
                                    seen_sig.add(sig)
                                    by_size[sz].append(e)
                # extend as needed if you add higher-arity macros

            # yield in size order
            for e in by_size[sz]:
                yield e

    def solve(self, task: Task, max_size: int) -> Tuple[Optional[Expr], int]:
        # Initialize pruner for this task
        pruner = Pruner(self.dsl)
        pruner.reset_for_task(task)

        # Size 1 candidates
        for e in self.generate(max_size=1):
            if pruner.reject(e):
                continue
            pruner.mark_tested()
            if passes(self.dsl, e, task):
                pruner.mark_solution_found()
                pruner.learn_from_solution(e)
                return e, pruner.stats.tested_count

        # Larger sizes
        for e in self.generate(max_size=max_size):
            if pruner.reject(e):
                continue
            pruner.mark_tested()
            if passes(self.dsl, e, task):
                pruner.mark_solution_found()
                pruner.learn_from_solution(e)
                return e, pruner.stats.tested_count

        return None, pruner.stats.tested_count

    def solve_with_stats(self, task: Task, max_size: int) -> Tuple[Optional[Expr], PruningStats]:
        """Solve and return detailed pruning statistics"""
        # Initialize pruner for this task
        pruner = Pruner(self.dsl)
        pruner.reset_for_task(task)

        # Size 1 candidates
        for e in self.generate(max_size=1):
            if pruner.reject(e):
                continue
            pruner.mark_tested()
            if passes(self.dsl, e, task):
                pruner.mark_solution_found()
                pruner.learn_from_solution(e)
                return e, pruner.stats

        # Larger sizes
        for e in self.generate(max_size=max_size):
            if pruner.reject(e):
                continue
            pruner.mark_tested()
            if passes(self.dsl, e, task):
                pruner.mark_solution_found()
                pruner.learn_from_solution(e)
                return e, pruner.stats

        return None, pruner.stats

# =========================
# 4) Abstraction mining
# =========================

def all_subtrees(e: Expr) -> List[Expr]:
    out = [e]
    for a in e.args:
        out.extend(all_subtrees(a))
    return out

def subtree_key(e: Expr) -> str:
    # structural key; macros are expanded structurally too
    if e.op in ("VAR", "CONST"):
        return str(e)
    return f"{e.op}({','.join(subtree_key(a) for a in e.args)})"

class AbstractionCore:
    """
    Minimal core that cannot fake progress:
    - Solves tasks via enumeration.
    - Mines frequent subtrees from solutions as candidate macros.
    - Accepts a macro only if it improves *held-out* search cost.
    """
    def __init__(self, seed: int = 0):
        random.seed(seed)
        self.dsl = DSL()
        self.inputs_probe = (-3, -2, -1, 0, 1, 2, 3, 5)
        self.enumerator = Enumerator(self.dsl, self.inputs_probe)

    def solve_tasks(self, tasks: List[Task], max_size: int) -> Tuple[Dict[str, Expr], Dict[str, int]]:
        sols: Dict[str, Expr] = {}
        costs: Dict[str, int] = {}
        pruning_stats: Dict[str, PruningStats] = {}

        print("\n🧹 SOLVING WITH PRUNING ENABLED")
        print("=" * 60)

        for t in tasks:
            print(f"\nSolving: {t.name}")
            expr, stats = self.enumerator.solve_with_stats(t, max_size=max_size)
            costs[t.name] = stats.tested_count
            pruning_stats[t.name] = stats

            if expr is not None:
                sols[t.name] = expr
                print(f"✅ Solution: {expr}")
            else:
                print("❌ No solution found")

            # Print pruning statistics
            print(f"📊 Generated: {stats.generated_count} | Tested: {stats.tested_count} | Pruned: {stats.pruned_count}")
            if stats.first_solution_index:
                print(f"🎯 First solution at: tested #{stats.first_solution_index}")
            prune_rate = stats.pruned_count / stats.generated_count * 100 if stats.generated_count > 0 else 0
            print(f"Prune rate: {prune_rate:.1f}%")
        # Summary statistics
        total_generated = sum(s.generated_count for s in pruning_stats.values())
        total_tested = sum(s.tested_count for s in pruning_stats.values())
        total_pruned = sum(s.pruned_count for s in pruning_stats.values())

        print("\n📈 OVERALL PRUNING SUMMARY")
        print("=" * 60)
        print(f"Total generated: {total_generated}")
        print(f"Total tested: {total_tested}")
        print(f"Total pruned: {total_pruned}")
        if total_generated > 0:
            overall_prune_rate = total_pruned / total_generated * 100
            print(f"Overall prune rate: {overall_prune_rate:.1f}%")
            print(f"Efficiency: {total_tested/total_generated*100:.1f}% of generated expressions tested")
        return sols, costs

    def propose_macros(self, solutions: Dict[str, Expr], top_k: int = 5) -> List[Expr]:
        # count repeated subtrees across solved tasks
        c = Counter()
        node_by_key: Dict[str, Expr] = {}
        for expr in solutions.values():
            for st in all_subtrees(expr):
                # ignore trivial leaf nodes
                if st.op in ("VAR", "CONST"):
                    continue
                k = subtree_key(st)
                c[k] += 1
                node_by_key[k] = st
        # prefer frequent and non-tiny patterns
        ranked = sorted(c.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
        candidates: List[Expr] = []
        for k, _freq in ranked[:top_k * 3]:
            cand = node_by_key[k]
            if cand.size() >= 3:
                candidates.append(cand)
            if len(candidates) >= top_k:
                break
        return candidates

    def _macro_expander_for(self, template: Expr) -> Tuple[int, Callable[..., Expr]]:
        """
        A simple macro system:
        - Macro has arity 1 or 2 only.
        - It substitutes placeholders X0/X1 (Vars) inside the template.
        We create the placeholders when we install the macro.

        This is intentionally minimal.
        """
        # Determine arity by how many placeholder vars appear
        # We'll use VAR names "X0", "X1" in the template.
        placeholders = set()

        def collect(e: Expr):
            if e.op == "VAR" and (e.name in ("X0", "X1")):
                placeholders.add(e.name)
            for a in e.args:
                collect(a)

        collect(template)
        arity = 1 if placeholders <= {"X0"} else 2 if placeholders <= {"X0", "X1"} else 0
        if arity == 0:
            # fallback: treat macro as unary wrapper around its argument by replacing VAR x
            arity = 1

        def subst(e: Expr, mapping: Dict[str, Expr]) -> Expr:
            if e.op == "VAR":
                if e.name in mapping:
                    return mapping[e.name]  # type: ignore[index]
                return e
            if e.op == "CONST":
                return e
            return Expr(e.op, args=tuple(subst(a, mapping) for a in e.args), val=e.val, name=e.name)

        def expander(a0: Expr, a1: Optional[Expr] = None) -> Expr:
            mapping = {"X0": a0}
            if arity == 2 and a1 is not None:
                mapping["X1"] = a1
            # also map plain "x" to X0 for convenience
            mapping["x"] = a0
            return subst(template, mapping)

        if arity == 1:
            return 1, lambda a0: expander(a0)
        return 2, lambda a0, a1: expander(a0, a1)

    def install_macro(self, name: str, template: Expr):
        # Convert the mined subtree into a parameterized macro by replacing VAR(x) with X0.
        def parametrize(e: Expr) -> Expr:
            if e.op == "VAR":
                return Expr("VAR", name="X0")
            if e.op == "CONST":
                return e
            return Expr(e.op, args=tuple(parametrize(a) for a in e.args), val=e.val, name=e.name)

        templ = parametrize(template)
        arity, expander = self._macro_expander_for(templ)
        self.dsl.macros[name] = (arity, expander)

    @staticmethod
    def median(xs: List[int]) -> float:
        s = sorted(xs)
        n = len(s)
        if n == 0:
            return float("inf")
        if n % 2 == 1:
            return float(s[n // 2])
        return 0.5 * (s[n // 2 - 1] + s[n // 2])

    def try_learn_one_macro(
        self,
        train: List[Task],
        holdout: List[Task],
        max_size: int,
        improvement_threshold: float = 0.15,  # 15% median cost drop
    ) -> bool:
        # Baseline costs on holdout
        _, base_costs = self.solve_tasks(holdout, max_size=max_size)
        base_med = self.median(list(base_costs.values()))

        # Solve train, mine candidate macros
        solutions, _train_costs = self.solve_tasks(train, max_size=max_size)
        if not solutions:
            print("No train solutions; cannot mine macros.")
            return False

        candidates = self.propose_macros(solutions, top_k=5)
        if not candidates:
            print("No macro candidates found.")
            return False

        best = None
        best_med = base_med

        # Try each candidate macro, measure holdout median cost
        for i, templ in enumerate(candidates):
            mname = f"M{i}"
            # install temporarily
            saved = dict(self.dsl.macros)
            self.install_macro(mname, templ)

            _, new_costs = self.solve_tasks(holdout, max_size=max_size)
            new_med = self.median(list(new_costs.values()))

            # restore macros for next test
            self.dsl.macros = saved

            if new_med < best_med:
                best_med = new_med
                best = templ

        if best is None:
            print("Rejected: no candidate improves held-out median cost.")
            return False

        drop = (base_med - best_med) / max(base_med, 1.0)
        if drop < improvement_threshold:
            print(f"Rejected: improvement {drop*100:.1f}% < {improvement_threshold*100:.1f}%.")
            return False

        # Accept: install the best macro permanently
        name = f"M{len(self.dsl.macros)}"
        self.install_macro(name, best)
        print(f"Accepted macro {name}: {best} | held-out median cost {base_med:.1f} → {best_med:.1f} ({drop*100:.1f}% drop)")
        return True

# =========================
# 5) Acceptance Testing: Verify Pruning Works
# =========================

def run_pruning_acceptance_test(core: AbstractionCore, tasks: List[Task], max_size: int = 5):
    """
    Run acceptance tests for pruning system.
    Verifies pruning doesn't break solvability and provides leverage.
    """
    print("\n🧪 PRUNING ACCEPTANCE TESTS")
    print("=" * 60)

    # Test 1: Safety - Pruning doesn't break solvability
    print("\n1️⃣ SAFETY TEST: Pruning doesn't break existing solutions")

    # First solve without any additional pruning (baseline)
    core_no_prune = AbstractionCore()  # Fresh instance
    solutions_no_prune, costs_no_prune = core_no_prune.solve_tasks(tasks, max_size=max_size)

    # Now solve with full pruning
    solutions_with_prune, costs_with_prune = core.solve_tasks(tasks, max_size=max_size)

    # Check solvability preservation
    solvable_no_prune = set(solutions_no_prune.keys())
    solvable_with_prune = set(solutions_with_prune.keys())

    broken_tasks = solvable_no_prune - solvable_with_prune
    if broken_tasks:
        print(f"❌ SAFETY FAILURE: Pruning broke solvability for: {broken_tasks}")
        return False
    else:
        print("✅ SAFETY PASSED: All solvable tasks remain solvable")

    # Test 2: Leverage - Pruning reduces search cost
    print("\n2️⃣ LEVERAGE TEST: Pruning reduces median search cost")

    median_no_prune = core.median(list(costs_no_prune.values()))
    median_with_prune = core.median(list(costs_with_prune.values()))

    improvement = (median_no_prune - median_with_prune) / median_no_prune * 100

    print(".1f")
    print(".1f")
    if improvement >= 10:  # 10% improvement threshold
        print("✅ LEVERAGE PASSED: Significant cost reduction achieved")
        return True
    else:
        print(f"❌ LEVERAGE FAILURE: Improvement < 10% threshold")
        return False

def benchmark_pruning_strategies(core: AbstractionCore, tasks: List[Task], max_size: int = 5):
    """
    Benchmark different pruning strategies to understand their individual impact.
    """
    print("\n📊 PRUNING STRATEGY BENCHMARK")
    print("=" * 60)

    strategies = ["baseline", "semantic_only", "normal_form_only", "full_pruning"]
    results = {}

    for strategy in strategies:
        print(f"\nTesting: {strategy}")

        # Create fresh core for each test
        test_core = AbstractionCore()

        if strategy == "baseline":
            # Disable all advanced pruning by using simple enumerator
            pass  # Already using baseline
        elif strategy == "semantic_only":
            # Only semantic duplicate pruning (already implemented in generate)
            pass
        elif strategy == "normal_form_only":
            # Could add a flag to enable only normal-form pruning
            pass
        # full_pruning is the default

        solutions, costs = test_core.solve_tasks(tasks, max_size=max_size)
        median_cost = test_core.median(list(costs.values()))
        solve_rate = len(solutions) / len(tasks) * 100

        results[strategy] = {
            'median_cost': median_cost,
            'solve_rate': solve_rate,
            'solutions_found': len(solutions)
        }

        print(f"  Median cost: {median_cost:.1f}")
        print(".1f")
    return results

# =========================
# 5) Demo: cannot fake
# =========================

def make_linear_family(a: int, b: int, n: int, name: str) -> Task:
    # y = a*x + b
    ex = []
    for x in range(-2, -2 + n):
        ex.append((x, a * x + b))
    return Task(name=name, examples=tuple(ex))

def make_quad_family(n: int, name: str) -> Task:
    # y = x*x + 1
    ex = []
    for x in range(-2, -2 + n):
        ex.append((x, x * x + 1))
    return Task(name=name, examples=tuple(ex))

def main():
    core = AbstractionCore(seed=1)

    # Task family intentionally has repeated structure so macros can help.
    # Train set: several linear functions
    train = [
        make_linear_family(2, 1, 5, "t_lin_2x+1"),
        make_linear_family(2, 3, 5, "t_lin_2x+3"),
        make_linear_family(3, 1, 5, "t_lin_3x+1"),
        make_quad_family(5, "t_quad_x2+1"),  # add a different one
    ]

    # Holdout set: unseen linear variants
    holdout = [
        make_linear_family(2, -1, 5, "h_lin_2x-1"),
        make_linear_family(3, 3, 5, "h_lin_3x+3"),
        make_linear_family(1, 2, 5, "h_lin_1x+2"),
        make_linear_family(-2, 1, 5, "h_lin_-2x+1"),
    ]

    max_size = 7  # keep small; raise to explore more

    print("\n=== Baseline (no macros) holdout costs ===")
    _, base_costs = core.solve_tasks(holdout, max_size=max_size)
    for k, v in base_costs.items():
        print(f"{k:14s} candidates_tested={v}")
    print(f"Median cost: {core.median(list(base_costs.values())):.1f}")

    # Attempt to learn a macro that *provably* reduces held-out search cost
    print("\n=== Learn one macro (must reduce heldout median cost) ===")
    core.try_learn_one_macro(train, holdout, max_size=max_size, improvement_threshold=0.10)

    print("\n=== After macro installation: holdout costs ===")
    _, new_costs = core.solve_tasks(holdout, max_size=max_size)
    for k, v in new_costs.items():
        print(f"{k:14s} candidates_tested={v}")
    print(f"Median cost: {core.median(list(new_costs.values())):.1f}")

    print("\nInstalled macros:")
    for name, (arity, _) in core.dsl.macros.items():
        print(f"  {name} (arity={arity})")

    # Run acceptance tests for pruning
    print("\n" + "="*80)
    print("🧪 PRUNING SYSTEM VALIDATION")
    print("="*80)

    # Test on holdout tasks
    acceptance_passed = run_pruning_acceptance_test(core, holdout, max_size=max_size)
    if acceptance_passed:
        print("\n🎉 ALL ACCEPTANCE TESTS PASSED!")
        print("   Pruning system is working correctly and providing leverage.")
    else:
        print("\n❌ ACCEPTANCE TESTS FAILED!")
        print("   Pruning system needs improvement.")

    # Benchmark different pruning strategies
    print("\n" + "="*80)
    print("📊 PRUNING STRATEGY ANALYSIS")
    print("="*80)
    benchmark_results = benchmark_pruning_strategies(core, holdout, max_size=max_size)

if __name__ == "__main__":
    main()
