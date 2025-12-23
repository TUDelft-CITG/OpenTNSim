from enum import Enum


class vessel_characteristics(Enum):
    min_ge_Length = ['minLength', '>=']
    min_gt_Length = ['minLength', '>']
    max_le_Length = ['maxLength', '<=']
    max_lt_Length = ['maxLength', '<']
    min_ge_Draught = ['minDraught', '>=']
    min_gt_Draught = ['minDraught', '>']
    max_le_Draught = ['maxDraught', '<=']
    max_lt_Draught = ['maxDraught', '<']
    min_ge_Beam = ['minBeam', '>=']
    min_gt_Beam = ['minBeam', '>']
    max_le_Beam = ['maxBeam', '<=']
    max_lt_Beam = ['maxBeam', '<']
    min_ge_UKC = ['minUKC', '>=']
    min_gt_UKC = ['minUKC', '>']
    max_le_UKC = ['maxUKC', '<=']
    max_lt_UKC = ['maxUKC', '<']
    type = ['Type', '==']
    terminal = ['Terminal','.isin(']
    visited_terminal = ['Previous terminal','.isin(']


class vessel_direction(Enum):
    inbound = 'inbound'
    outbound = 'outbound'


class vessel_type(Enum):
    GeneralCargo = 'GeneralCargo'
    LiquidBulk = 'LiquidBulk'
    Container = 'Container'
    DryBulk = 'DryBulk'
    MultiPurpose = 'MultiPurpose'
    Reefer = 'Reefer'
    RoRo = 'RoRo'
    Barge = 'Barge'


class MathematicalOperator(Enum):
    sum_of = "sum_of"
    max_of = "max_of"

    def apply(self, expr):
        return AggregateExpr(self, expr)


class MathematicalOperator(Enum):
    smaller_than = '<'

    def __call__(self):
        return Operator(self)


class VesselParameter(Enum):
    length = 'Vessel_length'
    beam = 'Vessel_beam'

    def __call__(self):
        return Parameter(self)


class WaterwayParameter(Enum):
    length = 'waterway_length'
    width = 'waterway_width'

    def __call__(self):
        return Parameter(self)


class Parameter:
    def __init__(self, param):
        self.param = param
        self.alias = param.value

    def render(self):
        return self.alias


class Operator:
    def __init__(self, symbol: MathematicalOperator):
        self.symbol = symbol
        self.alias = symbol.value

    def render(self):
        return self.alias


class Expr:
    def render(self):
        raise NotImplementedError


class ComparisonExpr(Expr):
    def __init__(self, left: Parameter, op: Operator, right: Parameter):
        self.left = left
        self.op = op
        self.right = right

    def render(self):
        return f"{self.left.render()} {self.op.render()} {self.right.render()}"


class AggregateExpr(ComparisonExpr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class IsRule:
    def __init__(self, expression, *args, **kwargs):
        self.expression = expression
        super().__init__(*args, **kwargs)
