import ast
import re
from typing import Optional


_PATTERNS = [
    (
        re.compile(r"what\s+is\s+(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)\s*\??", re.I),
        lambda m: _safe_eval(f"{m.group(1)}{m.group(2)}{m.group(3)}")
    ),
    (
        re.compile(r"what\s+is\s+(\d+\.?\d*)\s*%\s*of\s+(\d+\.?\d*)\s*\??", re.I),
        lambda m: str(float(m.group(1)) * float(m.group(2)) / 100.0)
    ),
    (
        re.compile(r"(\d+\.?\d*)\s*%\s*of\s+(\d+\.?\d*)", re.I),
        lambda m: str(float(m.group(1)) * float(m.group(2)) / 100.0)
    ),
    (
        re.compile(r"average\s+of\s+([\d,\s]+)", re.I),
        lambda m: _avg(m.group(1))
    ),
    (
        re.compile(r"mean\s+of\s+([\d,\s]+)", re.I),
        lambda m: _avg(m.group(1))
    ),
    (
        re.compile(r"(\d+\.?\d*)\s*[x\*\u00d7]\s*(\d+\.?\d*)"),
        lambda m: _safe_eval(f"{m.group(1)}*{m.group(2)}")
    ),
    (
        re.compile(r"(\d+\.?\d*)\s*[÷/]\s*(\d+\.?\d*)"),
        lambda m: str(float(m.group(1)) / float(m.group(2)))
    ),
    (
        re.compile(r"solve\s+x\s*([+\-])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)", re.I),
        lambda m: _solve_linear(m.group(1), m.group(2), m.group(3))
    ),
    (
        re.compile(r"(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*=\s*\?", re.I),
        lambda m: str(float(m.group(1)) + float(m.group(2)))
    ),
    (
        re.compile(r"if\s+(\d+\.?\d*)\s+(\w+)\s+cost\s+\$?(\d+\.?\d*)", re.I),
        lambda m: _proportional(m)
    ),
    (
        re.compile(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*=\s*\?", re.I),
        lambda m: str(float(m.group(1)) - float(m.group(2)))
    ),
    (
        re.compile(r"calculate\s+(\d+\.?\d*)\s*([+\-])\s*(\d+\.?\d*)\s*\.?", re.I),
        lambda m: _safe_eval(f"{m.group(1)}{m.group(2)}{m.group(3)}")
    ),
]


def _safe_eval(expr: str) -> str:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                     ast.Add, ast.Sub, ast.Mult, ast.Div,
                                     ast.Constant, ast.Num)):
                raise ValueError("unsafe expression")
        result = eval(expr, {"__builtins__": {}}, {})
        if isinstance(result, float):
            if result == int(result):
                return str(int(result))
            return str(result)
        return str(result)
    except Exception:
        raise


def _avg(nums_str: str) -> str:
    nums = [float(n.strip()) for n in nums_str.split(",") if n.strip()]
    if not nums:
        raise ValueError("no numbers")
    avg = sum(nums) / len(nums)
    return str(int(avg)) if avg == int(avg) else str(avg)


def _solve_linear(op: str, val: str, rhs: str) -> str:
    v = float(val)
    r = float(rhs)
    if op == "+":
        return str(int(r - v)) if (r - v) == int(r - v) else str(r - v)
    elif op == "-":
        return str(int(r + v)) if (r + v) == int(r + v) else str(r + v)
    return ""


def _proportional(m: re.Match) -> str:
    qty = float(m.group(1))
    cost = float(m.group(3))
    next_match = re.search(r"how\s+much\s+(?:for|do|are|is)\s+(\d+\.?\d*)", m.string, re.I)
    if next_match:
        next_qty = float(next_match.group(1))
        unit_price = cost / qty
        total = unit_price * next_qty
        return f"${int(total)}" if total == int(total) else f"${total:.2f}"
    return ""


def solve(prompt: str) -> tuple[Optional[str], Optional[float]]:
    lower = prompt.lower().strip()
    if not lower:
        return None, None
    for pattern, handler in _PATTERNS:
        m = pattern.search(lower)
        if m:
            try:
                result = handler(m)
                if result:
                    return result, 0.95
            except Exception:
                continue
    return None, None
