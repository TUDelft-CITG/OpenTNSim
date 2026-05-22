from dataclasses import dataclass
from typing import Callable, Any, List
from enum import Enum
import ast
import inspect

OPERATOR_MAP = {
    "Gt": ">",
    "Lt": "<",
    "GtE": ">=",
    "LtE": "<=",
    "Eq": "==",
    "NotEq": "!=",
}

class RuleParser(ast.NodeVisitor):

    def __init__(self, namespace=None):
        self.namespace = namespace or {}


    def visit_Call(self, node):
        from opentnsim.port.utils import extract_return_expr
        # any(...)
        if isinstance(node.func, ast.Name) and node.func.id == "any":

            return AnyOf(
                expr=self.visit(node.args[0])
            )

        # normal function call
        if isinstance(node.func, ast.Name):

            fn_name = node.func.id

            # try lambda globals
            fn = self.namespace.get(fn_name)

            # fallback: current module globals
            if fn is None:
                fn = globals().get(fn_name)

            # recursively inline restriction functions
            if inspect.isfunction(fn):

                try:
                    expr = extract_return_expr(fn)

                    # map function params -> actual args
                    params = fn.__code__.co_varnames[:fn.__code__.co_argcount]

                    args = [ast.unparse(a) for a in node.args]

                    mapping = dict(zip(params, args))

                    # substitute variables
                    expr = ParameterSubstituter(mapping).visit(expr)

                    ast.fix_missing_locations(expr)

                    return self.visit(expr)

                except Exception as e:
                    print(f"Could not expand {fn_name}: {e}")

        # fallback
        return Call(
            name=ast.unparse(node.func),
            args=[ast.unparse(a) for a in node.args]
        )


    def visit_GeneratorExp(self, node):
        return self.visit(node.elt)


    def visit_Compare(self, node):
        left = ast.unparse(node.left)

        op = type(node.ops[0]).__name__
        op = OPERATOR_MAP.get(op, op)

        right = ast.unparse(node.comparators[0])

        return Compare(
            left=left,
            op=op,
            right=right
        )


    def visit_BoolOp(self, node):

        values = [self.visit(v) for v in node.values]

        if isinstance(node.op, ast.And):
            return And(items=values)

        if isinstance(node.op, ast.Or):
            return Or(items=values)

        return values


class ParameterSubstituter(ast.NodeTransformer):

    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):

        if node.id in self.mapping:
            return ast.copy_location(
                ast.Name(id=self.mapping[node.id], ctx=node.ctx),
                node
            )

        return node
    

@dataclass
class Node:
    pass


@dataclass
class And(Node):
    items: List[Node]


@dataclass
class Or(Node):
    items: List[Node]


@dataclass
class Compare(Node):
    left: str
    op: str
    right: str


@dataclass
class Group(Node):
    expr: Node

@dataclass
class AnyOf(Node):
    expr: Any


@dataclass
class Call(Node):
    name: str
    args: list


@dataclass
class Rule:
    condition: Callable
    policy: Any


class RuleEngine:

    def __init__(self, default):
        self.rules = []
        self.default = default

    def add_rule(self, condition, policy):

        self.rules.append(
            Rule(
                condition=condition,
                policy=policy
            )
        )

    def evaluate(self, obj):

        for rule in self.rules:

            if rule.condition(obj):
                return rule.policy(obj)

        return self.default(obj)
    
    
    def overview(self):
        from opentnsim.port.utils import parse_rule, render_rule
        output = []

        for rule in self.rules:

            tree = parse_rule(rule.condition)

            output.append(
                f"{rule.condition.__name__}\n"
                f"{render_rule(tree)}"
            )

        print("\n\n".join(output))


class TrafficRules(Enum):

    prohibited = lambda v: 1
    allowed = lambda v: 0

    def __call__(self, vessel):
        return self.value(vessel)

