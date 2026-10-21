from dataclasses import dataclass
from typing import Callable, Any
from enum import Enum


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
        "Provides an overview of all the rules (conditions: policy) that are implemented"
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

