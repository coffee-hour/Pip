"""Specialists: exact math and Python syntax checks. Safe by construction, no eval()."""
import ast
import operator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def math_eval(expr: str):
    """Evaluate plain arithmetic like '(3 + 4) * 2 ** 3'. Raises ValueError on anything else."""
    return _eval(ast.parse(expr.strip(), mode="eval"))


def python_check(code: str) -> dict:
    """Return {'ok': bool, 'error': str|None} for a Python snippet. Parses only; never executes."""
    try:
        ast.parse(code)
        return {"ok": True, "error": None}
    except SyntaxError as e:
        return {"ok": False, "error": f"line {e.lineno}: {e.msg}"}
