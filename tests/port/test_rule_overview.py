"""Tests of the overview of the rules of a rule engine (RuleEngine.overview), which was broken by missing parser classes (issue #145)."""

import textwrap
from itertools import combinations

import pytest

from opentnsim.port.mixins.rules import MAX_INLINE_DEPTH, Rule, RuleEngine, TrafficRules, describe_policy
from opentnsim.port.utils import extract_return_expr, parse_rule, render_rule


def restriction_1(vessel_1, vessel_2):
    return (
        vessel_1.B >= 0.0
        and vessel_2.B >= 0.0
        and (vessel_1.classification != 'C' and vessel_2.classification != 'C')
    )


def restriction_2(vessel_1, vessel_2):
    return (
        (vessel_1.L > 340.0 or vessel_1.B > 49.0)
        and vessel_2.B >= 0.0
        and not (vessel_1.classification == 'C')
    )


def with_several_statements(vessel):
    if vessel.T > 5:
        return True
    return False


def with_docstring(vessel):
    """A restriction with a docstring."""
    return vessel.T > 5


def recursive(vessel):
    return recursive(vessel)


def overview(capsys, *rules):
    engine = RuleEngine(default=TrafficRules.allowed)
    for condition, policy, name in rules:
        engine.add_rule(condition, policy, name)
    engine.overview()
    return capsys.readouterr().out.strip()


def test_overview_expands_restriction_functions(capsys):
    output = overview(
        capsys,
        (
            lambda vessels: any(
                restriction_1(v1, v2)
                or restriction_1(v2, v1)
                for v1, v2 in combinations(vessels, 2)
            ),
            TrafficRules.prohibited,
            "5.1.1",
        ),
    )
    assert output == textwrap.dedent(
        """\
        5.1.1 (policy: prohibited)
          FOR ANY (v1, v2) in combinations(vessels, 2):
            ANY of:
              ALL of:
                v1.B >= 0.0
                v2.B >= 0.0
                v1.classification != 'C'
                v2.classification != 'C'
              ALL of:
                v2.B >= 0.0
                v1.B >= 0.0
                v2.classification != 'C'
                v1.classification != 'C'"""
    )


def test_overview_of_all_not_and_or(capsys):
    output = overview(
        capsys,
        (lambda vessels: all(restriction_2(v1, v2) for v1, v2 in combinations(vessels, 2)), TrafficRules.prohibited, "all"),
    )
    assert output == textwrap.dedent(
        """\
        all (policy: prohibited)
          FOR ALL (v1, v2) in combinations(vessels, 2):
            ALL of:
              ANY of:
                v1.L > 340.0
                v1.B > 49.0
              v2.B >= 0.0
              NOT v1.classification == 'C'"""
    )


def test_overview_of_a_condition_that_is_a_function(capsys):
    output = overview(capsys, (restriction_1, TrafficRules.allowed, "function"))
    assert output.splitlines()[:2] == ["function (policy: allowed)", "  ALL of:"]
    assert "    vessel_1.B >= 0.0" in output


def test_overview_with_arguments_by_keyword(capsys):
    output = overview(capsys, (lambda a, b: restriction_1(vessel_2=a, vessel_1=b), TrafficRules.prohibited, "keywords"))
    # vessel_1 is replaced by b and vessel_2 by a
    assert "    b.B >= 0.0\n    a.B >= 0.0" in output


def test_overview_chooses_the_right_lambda_from_a_line_with_two_lambdas(capsys):
    engine = RuleEngine(default=TrafficRules.allowed)
    engine.add_rule(lambda vessel: vessel.T >= 5.0, lambda vessel: vessel.T * 2, "two lambdas")
    engine.overview()
    assert capsys.readouterr().out.strip() == "two lambdas (policy: vessel.T * 2)\n  vessel.T >= 5.0"


def test_overview_of_a_lambda_that_is_part_of_a_larger_statement(capsys):
    # the source of the lambda is the line of the dictionary entry, which cannot be interpreted on its own
    restrictions = {
        ("a", "b"): [Rule(condition=lambda vessel: vessel.T >= 5.0, policy=TrafficRules.prohibited, name="draught")],
        ("c", "d"): [Rule(condition=lambda vessel: vessel.L > 100.0 and vessel.B > 20.0, policy=TrafficRules.allowed, name="size")],
    }
    engine = RuleEngine(default=TrafficRules.allowed)
    for rules in restrictions.values():
        for rule in rules:
            engine.add_rule(rule.condition, rule.policy, rule.name)
    engine.overview()
    assert capsys.readouterr().out.strip() == textwrap.dedent(
        """\
        draught (policy: prohibited)
          vessel.T >= 5.0

        size (policy: allowed)
          ALL of:
            vessel.L > 100.0
            vessel.B > 20.0"""
    )


def test_functions_with_several_statements_are_not_expanded(capsys):
    output = overview(capsys, (lambda vessel: with_several_statements(vessel) and not vessel.loaded, TrafficRules.prohibited, "calls"))
    assert output.splitlines()[1:] == ["  ALL of:", "    with_several_statements(vessel)", "    NOT vessel.loaded"]


def test_functions_with_a_docstring_are_expanded(capsys):
    output = overview(capsys, (lambda vessel: with_docstring(vessel), TrafficRules.prohibited, "docstring"))
    assert output.splitlines()[1:] == ["  vessel.T > 5"]


def test_recursive_functions_are_expanded_up_to_a_maximum_depth(capsys):
    output = overview(capsys, (lambda vessel: recursive(vessel), TrafficRules.prohibited, "recursive"))
    assert output.splitlines()[1:] == ["  recursive(vessel)"]
    assert MAX_INLINE_DEPTH == 10


def test_overview_without_source(capsys):
    # the source of a builtin is not available
    output = overview(capsys, (len, TrafficRules.allowed, "builtin"))
    assert output == "builtin (policy: allowed)\n  len (source not available)"


def test_overview_without_rules(capsys):
    assert overview(capsys) == ""


# %% parse_rule, render_rule, extract_return_expr and describe_policy
def test_parse_and_render_rule():
    tree = parse_rule(lambda vessel: vessel.T > 5 and not vessel.loaded)
    assert render_rule(tree) == "ALL of:\n  vessel.T > 5\n  NOT vessel.loaded"
    assert render_rule(tree, indent=1).startswith("  ALL of:")


def test_extract_return_expr_requires_a_single_return_statement():
    with pytest.raises(ValueError, match="single return statement"):
        extract_return_expr(with_several_statements)


def policy_function(vessel):
    return 0


@pytest.mark.parametrize(
    "policy, expected",
    [
        (TrafficRules.prohibited, "prohibited"),
        (TrafficRules.allowed, "allowed"),
        (policy_function, "policy_function"),
        (lambda vessel: vessel.T * 2, "vessel.T * 2"),
        (len, "len"),
    ],
)
def test_describe_policy(policy, expected):
    assert describe_policy(policy) == expected
