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

# the kinds of function parameters that can be given an argument by position
POSITIONAL_KINDS = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

# maximum number of nested restriction functions that is expanded in the overview of a rule
MAX_INLINE_DEPTH = 10

# the indentation of one level in the overview of a rule
INDENT = "  "


class Node:
    """Parent class of the nodes in the tree of a rule condition (see RuleParser).

    The tree is shown by RuleEngine.overview.
    """

    def render(self, indent=0):
        """Text of the node (and its children), indented with two spaces per level"""
        raise NotImplementedError


@dataclass
class Expression(Node):
    """An expression that is not split up further, e.g. 'vessel.T > 10' or 'vessel.is_loaded'."""

    text: str

    def render(self, indent=0):
        return INDENT * indent + self.text


@dataclass
class Compare(Node):
    left: str
    op: str
    right: str

    def render(self, indent=0):
        return INDENT * indent + f"{self.left} {self.op} {self.right}"


@dataclass
class Call(Node):
    """A call of a function that is not expanded, e.g. a function from a library or a function with several statements."""

    name: str
    args: list

    def render(self, indent=0):
        arguments = ", ".join(self.args)
        return INDENT * indent + f"{self.name}({arguments})"


@dataclass
class Not(Node):
    item: Node

    def render(self, indent=0):
        pad = INDENT * indent
        if isinstance(self.item, (Expression, Compare, Call)):
            # short enough to be on the same line
            return f"{pad}NOT {self.item.render()}"
        lines = [f"{pad}NOT:", self.item.render(indent + 1)]
        return "\n".join(lines)


@dataclass
class And(Node):
    items: List[Node]

    def render(self, indent=0):
        lines = [INDENT * indent + "ALL of:"]
        for item in self.items:
            lines.append(item.render(indent + 1))
        return "\n".join(lines)


@dataclass
class Or(Node):
    items: List[Node]

    def render(self, indent=0):
        lines = [INDENT * indent + "ANY of:"]
        for item in self.items:
            lines.append(item.render(indent + 1))
        return "\n".join(lines)


@dataclass
class AnyOf(Node):
    """any(expr for ... in ...): the condition holds if the expression holds for at least one of the items."""

    expr: Node
    iteration: str

    def render(self, indent=0):
        lines = [INDENT * indent + f"FOR ANY {self.iteration}:", self.expr.render(indent + 1)]
        return "\n".join(lines)


@dataclass
class AllOf(Node):
    """all(expr for ... in ...): the condition holds if the expression holds for all of the items."""

    expr: Node
    iteration: str

    def render(self, indent=0):
        lines = [INDENT * indent + f"FOR ALL {self.iteration}:", self.expr.render(indent + 1)]
        return "\n".join(lines)


class ParameterSubstituter(ast.NodeTransformer):
    """Replaces the names of function parameters in an expression by the expressions that were passed as arguments.

    For the function
        def restriction(vessel_1, vessel_2):
            return vessel_1.B > 0 and vessel_2.B > 0
    that is called as restriction(v2, v1), the mapping is {'vessel_1': v2, 'vessel_2': v1} and the expression
    'vessel_1.B > 0 and vessel_2.B > 0' becomes 'v2.B > 0 and v1.B > 0'.

    Parameters
    ----------
    mapping : dict
        the name of each parameter and the syntax tree of the argument that is passed for it
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id not in self.mapping:
            return node
        # a copy, because a parameter can be used several times in the expression and the nodes of the syntax tree
        # should not be shared
        return copy.deepcopy(self.mapping[node.id])


class RuleParser(ast.NodeVisitor):
    """Translates the expression of a rule condition into a tree of nodes.

    The nodes are And, Or, Not, Compare, AnyOf, AllOf, Call and Expression.

    Calls of functions that consist of a single return statement (restriction functions) are replaced by the
    expression of that function, with the parameters replaced by the arguments of the call. Other expressions are
    kept as text.

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
        # ast.NodeVisitor calls this for every type of node that has no visit_<Type> method (names, attributes,
        # constants, arithmetic, ...). Such an expression is not split up further, but shown as text.
        return Expression(ast.unparse(node))

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            node_type = And
        else:
            node_type = Or

        items = []
        for value in node.values:
            item = self.visit(value)
            if isinstance(item, node_type):
                # a nested expression with the same operator is part of the same list: (a and b) and c -> ALL of: a, b, c
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

        operator_type = type(node.ops[0])
        operator = OPERATOR_MAP.get(operator_type, operator_type.__name__)
        left = ast.unparse(node.left)
        right = ast.unparse(node.comparators[0])
        return Compare(left=left, op=operator, right=right)

    def visit_Call(self, node):
        # A call is either (1) any(...) or all(...), which is shown with its loop, (2) a call of a restriction function
        # in the namespace, which is replaced by the expression of that function, or (3) any other call, which is shown
        # as a call. Only calls of a name are expanded: for 'module.function(...)' the function is not looked up.
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
            one_argument = len(node.args) == 1 and not node.keywords
            if function_name in ("any", "all") and one_argument:
                quantifier = self._visit_any_or_all(node)
                if quantifier is not None:
                    return quantifier

            expanded = self._expand_function(node)
            if expanded is not None:
                return expanded

        return self._call_as_text(node)

    @staticmethod
    def _call_as_text(node):
        """A call that is shown as text: the name of the function and its arguments"""
        arguments = []
        for argument in node.args:
            arguments.append(ast.unparse(argument))
        for keyword in node.keywords:
            arguments.append(ast.unparse(keyword))
        return Call(name=ast.unparse(node.func), args=arguments)

    def _visit_any_or_all(self, node):
        """any(...) or all(...) over a generator or a list: None if the argument is of another type."""
        is_any = node.func.id == "any"
        argument = node.args[0]

        # any(<expression> for <loops>): the expression is shown once, for 'any' or 'all' of the items of the loops
        if isinstance(argument, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            iteration = self._describe_iteration(argument.generators)
            expression = self.visit(argument.elt)
            if is_any:
                return AnyOf(expr=expression, iteration=iteration)
            return AllOf(expr=expression, iteration=iteration)

        # any([a, b, c]): the same as 'a or b or c'
        if isinstance(argument, (ast.List, ast.Tuple, ast.Set)):
            items = [self.visit(element) for element in argument.elts]
            if is_any:
                return Or(items=items)
            return And(items=items)

        return None

    @staticmethod
    def _describe_iteration(generators):
        """The loops of a generator expression as text, e.g. 'v1 in vessels if v1.T > 5'"""
        loops = []
        for generator in generators:
            target = ast.unparse(generator.target)
            iterable = ast.unparse(generator.iter)
            loop = f"{target} in {iterable}"
            for condition in generator.ifs:
                loop += f" if {ast.unparse(condition)}"
            loops.append(loop)
        return ", ".join(loops)

    def _expand_function(self, node):
        """The expression of a restriction function with the parameters replaced by the arguments, or None if not possible."""
        # Example: in the condition 'restriction(v2, v1)', with
        #     def restriction(vessel_1, vessel_2):
        #         return vessel_1.B > 0 and vessel_2.B > 0
        # the call is replaced by 'v2.B > 0 and v1.B > 0'. The steps are:
        #   1. find the function by its name
        #   2. match the arguments of the call to the parameters of the function
        #   3. get the expression that the function returns (from its source)
        #   4. replace the parameters in that expression by the arguments
        #   5. translate that expression to a tree, as for the condition itself (it can call other restriction functions)
        from opentnsim.port.utils import extract_return_expr

        # 1. only functions that are defined in Python can be expanded (not builtins, or objects that are called)
        function = self.namespace.get(node.func.id)
        if not inspect.isfunction(function):
            return None

        # a function that (indirectly) calls itself would be expanded forever
        if self.depth >= MAX_INLINE_DEPTH:
            return None

        # 2.
        mapping = self._bind_arguments(function, node)
        if mapping is None:
            return None

        # 3.
        try:
            expression = extract_return_expr(function)
        except (OSError, TypeError, SyntaxError, ValueError):
            # the source of the function is not available, or the function is not a single return statement
            return None

        # 4. on a copy: the syntax tree of the function is also used when the function is called again, with other arguments
        expression = copy.deepcopy(expression)
        expression = ParameterSubstituter(mapping).visit(expression)

        # 5. the names in the expression are looked up in the namespace of the function that it came from, not in the
        # namespace of the condition: the function can be defined in another module
        parser = RuleParser(namespace=function_namespace(function), depth=self.depth + 1)
        return parser.visit(expression)

    @staticmethod
    def _bind_arguments(function, node):
        """The argument expression for each parameter of a call, or None if the call does not match the parameters.

        For restriction(a, vessel_2=b) with the parameters (vessel_1, vessel_2) this is {'vessel_1': a, 'vessel_2': b}.
        """
        # restriction(*vessels) and restriction(**options) cannot be matched to parameters without running the code
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                return None
        for keyword in node.keywords:
            if keyword.arg is None:
                # **kwargs
                return None

        # the parameters that can be given by position, in the order of the definition. Parameters that can only be given
        # by keyword, *args and **kwargs do not take positional arguments.
        parameters = []
        for name, parameter in inspect.signature(function).parameters.items():
            if parameter.kind in POSITIONAL_KINDS:
                parameters.append(name)
        if len(node.args) > len(parameters):
            return None

        # the first argument is for the first parameter, etc.; then the arguments that are given by keyword. Parameters
        # without an argument (that have a default value) are not in the mapping: their name stays in the expression.

        mapping = dict(zip(parameters, node.args))
        for keyword in node.keywords:
            if keyword.arg not in parameters:
                return None
            mapping[keyword.arg] = keyword.value
        return mapping


def function_namespace(function):
    """The names that are available to a function: its globals and the variables of the function(s) it is defined in."""
    # The globals are the names of the module in which the function is defined. For a function that is defined inside
    # another function (e.g. a lambda in a function that creates rules) the names of that function are not globals,
    # but 'free variables': their names are in co_freevars and their values in the cells of the closure, in the same order.
    namespace = dict(function.__globals__)
    names = function.__code__.co_freevars
    cells = function.__closure__ or ()
    for name, cell in zip(names, cells):
        try:
            namespace[name] = cell.cell_contents
        except ValueError:
            # empty cell: a variable that is not assigned yet
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
            policy = describe_policy(rule.policy)
            header = f"{rule.name} (policy: {policy})"
            try:
                tree = parse_rule(rule.condition)
                condition = render_rule(tree, indent=1)
            except (OSError, TypeError, SyntaxError, ValueError):
                # the source of the condition is not available (e.g. a function that is defined interactively)
                condition_name = getattr(rule.condition, "__name__", repr(rule.condition))
                condition = f"{INDENT}{condition_name} (source not available)"
            output.append(f"{header}\n{condition}")

        print("\n\n".join(output))


def describe_policy(policy):
    """Short text for the policy of a rule

    TrafficRules.prohibited -> 'prohibited', a function -> its name, a lambda -> its expression
    """
    from opentnsim.port.utils import extract_return_expr

    # TrafficRules.prohibited is not an Enum member but a plain function: Enum does not turn functions that are defined in
    # the class into members. Its name is 'lambda', so find the name that it has in the class instead.
    for name, member in vars(TrafficRules).items():
        if member is policy and not name.startswith("_"):
            return name

    name = getattr(policy, "__name__", None)
    if name and name != "<lambda>":
        return name

    try:
        expression = extract_return_expr(policy)
    except (OSError, TypeError, SyntaxError, ValueError):
        return repr(policy)
    return ast.unparse(expression)


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

