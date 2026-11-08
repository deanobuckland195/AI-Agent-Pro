"""Small arithmetic AST interpreter; no names, calls, attributes, or code evaluation."""
import ast
import math
import operator

from ai_agent_pro.tools.contracts import ToolFailure

OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
             ast.Pow: operator.pow}


def run(arguments, context):
    try:
        tree = ast.parse(arguments['expression'], mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError('AST size')

        def calculate(node, depth=0):
            if depth > 16:
                raise ValueError('AST depth')
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                result = node.value
            elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
                value = calculate(node.operand, depth + 1)
                result = value if isinstance(node.op, ast.UAdd) else -value
            elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
                left, right = calculate(node.left, depth + 1), calculate(node.right, depth + 1)
                if isinstance(node.op, ast.Pow) and (abs(right) > 12 or (left < 0 and int(right) != right)):
                    raise ValueError('Exponent limit')
                result = OPERATORS[type(node.op)](left, right)
            else:
                raise ValueError('Unsupported syntax')
            if type(result) not in (int, float) or not math.isfinite(result) or abs(result) > 1e100:
                raise ValueError('Number limit')
            return result

        return {'value': calculate(tree.body)}
    except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
        raise ToolFailure('invalid_expression') from None
