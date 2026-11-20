import ast
import copy
import inspect
from dataclasses import dataclass
from typing import Callable, Any, List
from enum import Enum


OPERATOR_MAP = {
    ast.Gt: ">",
    ast.Lt: "<",
    ast.GtE: ">=",
    ast.LtE: "<=",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
}

# maximum number of nested restriction functions that is expanded in the overview of a rule
MAX_INLINE_DEPTH = 10


class Node:
    """Parent class of the nodes in the tree of a rule condition (see RuleParser). The tree is shown by RuleEngine.overview."""

    def render(self, indent=0):
        """Text of the node (and its children), indented with two spaces per level"""
        raise NotImplementedError


@dataclass
class Expression(Node):
    """An expression that is not split up further, e.g. 'vessel.T > 10' or 'vessel.is_loaded'."""

    text: str

    def render(self, indent=0):
        return "  " * indent + self.text


@dataclass
class Compare(Node):
    left: str
    op: str
    right: str

    def render(self, indent=0):
        return "  " * indent + f"{self.left} {self.op} {self.right}"


@dataclass
class Call(Node):
    """A call of a function that is not expanded, e.g. a function from a library or a function with several statements."""

    name: str
    args: list

    def render(self, indent=0):
        return "  " * indent + f"{self.name}({', '.join(self.args)})"


@dataclass
class Not(Node):
    item: Node

    def render(self, indent=0):
        pad = "  " * indent
        if isinstance(self.item, (Expression, Compare, Call)):
            return f"{pad}NOT {self.item.render().strip()}"
        return "\n".join([f"{pad}NOT:", self.item.render(indent + 1)])


@dataclass
class And(Node):
    items: List[Node]

    def render(self, indent=0):
        return "\n".join(["  " * indent + "ALL of:"] + [item.render(indent + 1) for item in self.items])


@dataclass
class Or(Node):
    items: List[Node]

    def render(self, indent=0):
        return "\n".join(["  " * indent + "ANY of:"] + [item.render(indent + 1) for item in self.items])


@dataclass
class AnyOf(Node):
    """any(expr for ... in ...): the condition holds if the expression holds for at least one of the items."""

    expr: Node
    iteration: str

    def render(self, indent=0):
        return "\n".join(["  " * indent + f"FOR ANY {self.iteration}:", self.expr.render(indent + 1)])


@dataclass
class AllOf(Node):
    """all(expr for ... in ...): the condition holds if the expression holds for all of the items."""

    expr: Node
    iteration: str

    def render(self, indent=0):
        return "\n".join(["  " * indent + f"FOR ALL {self.iteration}:", self.expr.render(indent + 1)])


class ParameterSubstituter(ast.NodeTransformer):
    """Replaces the names of function parameters in an expression by the expressions that were passed as arguments."""

    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id in self.mapping:
            return copy.deepcopy(self.mapping[node.id])
        return node


class RuleParser(ast.NodeVisitor):
    """Translates the expression of a rule condition into a tree of nodes (And, Or, Not, Compare, AnyOf, AllOf, Call, Expression).

    Calls of functions that consist of a single return statement (restriction functions) are replaced by the expression
    of that function, with the parameters replaced by the arguments of the call. Other expressions are kept as text.

    Parameters
    ----------
    namespace : dict
        the names that can be used in the expression (to find the restriction functions), e.g. the globals of the condition
    depth : int
        the number of restriction functions in which this expression is nested
    """

    def __init__(self, namespace=None, depth=0):
        self.namespace = namespace or {}
        self.depth = depth

    def generic_visit(self, node):
        return Expression(ast.unparse(node))

    def visit_BoolOp(self, node):
        node_type = And if isinstance(node.op, ast.And) else Or
        items = []
        for value in node.values:
            item = self.visit(value)
            # a nested expression with the same operator is part of the same list: (a and b) and c -> ALL of: a, b, c
            if isinstance(item, node_type):
                items.extend(item.items)
            else:
                items.append(item)
        return node_type(items=items)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            return Not(item=self.visit(node.operand))
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if len(node.ops) != 1:
            # chained comparison: a < b < c
            return self.generic_visit(node)
        return Compare(
            left=ast.unparse(node.left),
            op=OPERATOR_MAP.get(type(node.ops[0]), type(node.ops[0]).__name__),
            right=ast.unparse(node.comparators[0]),
        )

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in ("any", "all") and len(node.args) == 1 and not node.keywords:
                quantifier = self._visit_any_or_all(node)
                if quantifier is not None:
                    return quantifier
            expanded = self._expand_function(node)
            if expanded is not None:
                return expanded
        arguments = [ast.unparse(arg) for arg in node.args] + [ast.unparse(keyword) for keyword in node.keywords]
        return Call(name=ast.unparse(node.func), args=arguments)

    def _visit_any_or_all(self, node):
        """any(...) or all(...) over a generator or a list: None if the argument is of another type."""
        any_ = node.func.id == "any"
        argument = node.args[0]
        if isinstance(argument, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            iteration = ", ".join(
                f"{ast.unparse(generator.target)} in {ast.unparse(generator.iter)}"
                + "".join(f" if {ast.unparse(condition)}" for condition in generator.ifs)
                for generator in argument.generators
            )
            return (AnyOf if any_ else AllOf)(expr=self.visit(argument.elt), iteration=iteration)
        if isinstance(argument, (ast.List, ast.Tuple, ast.Set)):
            return (Or if any_ else And)(items=[self.visit(element) for element in argument.elts])
        return None

    def _expand_function(self, node):
        """The expression of a restriction function with the parameters replaced by the arguments, or None if not possible."""
        from opentnsim.port.utils import extract_return_expr

        function = self.namespace.get(node.func.id)
        if not inspect.isfunction(function) or self.depth >= MAX_INLINE_DEPTH:
            return None
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(keyword.arg is None for keyword in node.keywords):
            return None
        parameters = [
            name
            for name, parameter in inspect.signature(function).parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if len(node.args) > len(parameters) or any(name not in parameters for name in keywords):
            return None
        try:
            expression = extract_return_expr(function)
        except (OSError, TypeError, SyntaxError, ValueError):
            return None
        mapping = dict(zip(parameters, node.args))
        mapping.update(keywords)
        expression = ParameterSubstituter(mapping).visit(copy.deepcopy(expression))
        return RuleParser(namespace=function_namespace(function), depth=self.depth + 1).visit(expression)


def function_namespace(function):
    """The names that are available to a function: its globals and the variables of the function(s) it is defined in."""
    namespace = dict(function.__globals__)
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
        try:
            namespace[name] = cell.cell_contents
        except ValueError:
            # empty cell: variable that is not assigned yet
            pass
    return namespace


@dataclass
class Rule:
    """Dataclass: Represent a rule consisting of a condition, a policy, and a name."""
    condition: Callable
    policy: Any
    name: str


class RuleEngine:
    """Class: Manages and evaluates rules associated with edges and nodes
        
    Used to implement traffic rules and tidal regulations

    Parameters
    ----------
    default: TrafficRules.prohibited or TrafficRules.allowed
        default value of the rule: prohibited (lambda v: 1) or allowed (lambda v: 0)

    Attributes:
    rules: list
        a list to which Rule mixins can be added
    default: TrafficRules.prohibited or TrafficRules.allowed
        default value of the rule: prohibited (lambda v: 1) or allowed (lambda v: 0)
    """
    def __init__(self, default):
        """Initialization"""
        self.rules = []
        self.default = default

    def add_rule(self, condition, policy, name):
        """Adds rule to a list of rules
                                                
        Input
        -----
        condition: callable
            predicate to evaluate, returning True when the rule's condition is satisfied (the input may contain one or more objects, such as a list of two vessels)
        policy: TrafficRules.prohibited or TrafficRules.allowed
            policy to apply when the condition is satisfied: TrafficRules.prohibited or TrafficRules.allowed.
        name: str
            descriptive name of the rule
        """
        self.rules.append(
            Rule(
                condition=condition,
                policy=policy,
                name=name,
            )
        )

    def evaluate(self, obj):
        """Evaluate all rules against the given object.

        Each rule's condition is evaluated against a object (obj). 
        If the condition is satisfied, the rule's policy is applied; otherwise, the default policy is applied.

        Parameters
        ----------
        obj : Any
            object or collection of objects passed to the rule conditions and policies.

        Returns
        -------
        rules: dict [str, int]
            dictionary mapping each rule's name to the result of its policy if its condition is satisfied, or to the result of the default policy otherwise.
        """
        rules = {}
        for rule in self.rules:
            if rule.condition(obj):
                rules[rule.name] = rule.policy(obj)
            else:
                rules[rule.name] = self.default(obj)
        return rules
    
    def overview(self):
        """Prints an overview of all the rules: the name, the policy and the condition as a tree

        Restriction functions that are called in a condition are expanded, for example::

            5.1.1 (policy: prohibited)
              FOR ANY (v1, v2) in combinations(vessels, 2):
                ANY of:
                  ALL of:
                    v1.B >= 0.0
                    v2.classification != 'C'
                  ALL of:
                    v2.B >= 0.0
                    v1.classification != 'C'
        """
        from opentnsim.port.utils import parse_rule, render_rule

        output = []
        for rule in self.rules:
            try:
                condition = render_rule(parse_rule(rule.condition), indent=1)
            except (OSError, TypeError, SyntaxError, ValueError):
                # the source of the condition is not available (e.g. a function that is defined interactively)
                condition = f"  {getattr(rule.condition, '__name__', repr(rule.condition))} (source not available)"
            output.append(f"{rule.name} (policy: {describe_policy(rule.policy)})\n{condition}")

        print("\n\n".join(output))


def describe_policy(policy):
    """Short text for the policy of a rule: TrafficRules.prohibited -> 'prohibited', a function -> its name, a lambda -> its expression"""
    from opentnsim.port.utils import extract_return_expr

    for name, member in vars(TrafficRules).items():
        if member is policy and not name.startswith("_"):
            return name
    name = getattr(policy, "__name__", None)
    if name and name != "<lambda>":
        return name
    try:
        return ast.unparse(extract_return_expr(policy))
    except (OSError, TypeError, SyntaxError, ValueError):
        return repr(policy)


class TrafficRules(Enum):
    """Enum class: Defines the default traffic rule behaviors.

    Each member represents a rule that determines whether a vessel is prohibited or allowed to pass. 
    Enum members are callable and evaluate the associated function for a given vessel.

    Attributes
    ----------
    prohibited: TrafficRules
        rule that prohibits passage -> returns 1 for any vessel
    allowed: TrafficRules
        rule that allows passage -> returns 0 for any vessel

    Methods
    -------
    __call__:
        evaluates the rule for the given vessel and returns the result
    """

    prohibited = lambda v: 1
    allowed = lambda v: 0

    def __call__(self, vessel):
        return self.value(vessel)

