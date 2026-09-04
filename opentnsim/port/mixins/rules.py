from dataclasses import dataclass
from typing import Callable, Any
from enum import Enum


@dataclass
class Rule:
    condition: Callable
    policy: Any
    name: str


class RuleEngine:

    def __init__(self, default):
        self.rules = []
        self.default = default

    def add_rule(self, condition, policy, name):

        self.rules.append(
            Rule(
                condition=condition,
                policy=policy,
                name=name,
            )
        )

    def evaluate(self, obj):
        rules = {}
        for rule in self.rules:
            if rule.condition(obj):
                rules[rule.name] = rule.policy(obj)
            else:
                rules[rule.name] = self.default(obj)
        return rules
    
    
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

