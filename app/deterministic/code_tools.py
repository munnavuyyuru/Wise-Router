import ast
import re
from typing import Optional


def _detect_mutable_defaults(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fixes = []
    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            for arg in node.args.defaults:
                if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                    fixes.append(
                        f"{node.name}: mutable default {ast.dump(arg)} "
                        f"→ use None + None-check inside function"
                    )
            for d in node.args.kw_defaults:
                if d is not None and isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    fixes.append(
                        f"{node.name}: mutable kw default {ast.dump(d)}"
                    )
    Visitor().visit(tree)
    if fixes:
        return "; ".join(fixes)
    return None


def _detect_missing_return(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fixes = []
    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            has_return = False
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    has_return = True
                    break
            if not has_return and node.name != "__init__":
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id in ("property", "abstractmethod"):
                        return
                body_ends_with_expr = node.body and isinstance(node.body[-1], ast.Expr)
                if body_ends_with_expr:
                    fixes.append(f"{node.name}: last statement may need 'return' keyword")
    Visitor().visit(tree)
    if fixes:
        return "; ".join(fixes)
    return None


def _detect_off_by_one(code: str) -> Optional[str]:
    pattern = re.compile(r"for\s+\w+\s+in\s+range\(len\((\w+)\)\):")
    matches = pattern.findall(code)
    if not matches:
        return None
    issues = []
    for var in matches:
        if f"{var}[i+1]" in code or f"{var}[{var}+1]" in code:
            issues.append(f"potential off-by-one: iterating {var} indexes, accessing i+1")
    return "; ".join(issues) if issues else None


def _detect_recursion_base(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fixes = []
    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            calls_self = any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == node.name
                for child in ast.walk(node)
            )
            if calls_self:
                has_base = any(
                    isinstance(child, ast.If)
                    and any(
                        isinstance(s, ast.Return)
                        and (s.value is None or not isinstance(s.value, ast.Call))
                        for s in ast.walk(child)
                    )
                    for child in ast.walk(node)
                )
                if not has_base:
                    fixes.append(f"{node.name}: recursive but no explicit base-case return found")
    Visitor().visit(tree)
    if fixes:
        return "; ".join(fixes)
    return None


def _detect_assignment_in_condition(code: str) -> Optional[str]:
    pattern = re.compile(r'(?:if|while|elif)\s+\w+\s*=\s*\w+', re.MULTILINE)
    matches = pattern.findall(code)
    if matches:
        return "assignment (=) used in condition; did you mean ==?"
    return None


def _detect_missing_colon(code: str) -> Optional[str]:
    lines = code.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "#" in stripped:
            stripped = stripped.split("#")[0].strip()
        if not stripped:
            continue
        if re.match(r'^(if |elif |else|for |while |def |class |try|except|finally)', stripped):
            if not stripped.rstrip().endswith(":"):
                return f"line {i + 1}: missing colon after '{stripped.strip()}'"
    return None


TEMPLATES = {
    "sort": (
        "def sort_array(arr):\n"
        "    return sorted(arr)\n"
    ),
    "reverse": (
        "def reverse_string(s):\n"
        "    return s[::-1]\n"
    ),
    "factorial": (
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n"
    ),
    "fibonacci": (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    ),
    "palindrome": (
        "def is_palindrome(s):\n"
        "    s = s.lower().replace(' ', '')\n"
        "    return s == s[::-1]\n"
    ),
    "is_prime": (
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, int(n**0.5) + 1):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n"
    ),
    "fizzbuzz": (
        "def fizzbuzz(n):\n"
        "    result = []\n"
        "    for i in range(1, n + 1):\n"
        "        if i % 15 == 0:\n"
        "            result.append('FizzBuzz')\n"
        "        elif i % 3 == 0:\n"
        "            result.append('Fizz')\n"
        "        elif i % 5 == 0:\n"
        "            result.append('Buzz')\n"
        "        else:\n"
        "            result.append(str(i))\n"
        "    return result\n"
    ),
    "gcd": (
        "def gcd(a, b):\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return a\n"
    ),
}

TEMPLATE_KEYWORDS = {
    "sort": ["sort", "order", "ascending", "descending"],
    "reverse": ["reverse", "backward"],
    "factorial": ["factorial"],
    "fibonacci": ["fibonacci", "fib"],
    "palindrome": ["palindrome"],
    "is_prime": ["prime", "is_prime", "primality"],
    "fizzbuzz": ["fizzbuzz", "fizz", "buzz"],
    "gcd": ["gcd", "greatest common divisor", "hcf"],
}


def _extract_code(text: str) -> str:
    lines = text.splitlines()
    code_lines = []
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_lines.append(line)
            continue
        if "def " in stripped or "class " in stripped or "return " in stripped:
            code_lines.append(line)
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            code_lines.append(line)
            continue
        if stripped.startswith("try:") or stripped.startswith("except"):
            code_lines.append(line)
            continue
        if stripped.startswith("if ") or stripped.startswith("for ") or stripped.startswith("while "):
            code_lines.append(line)
            continue
        if "=" in stripped and not stripped.startswith("#"):
            code_lines.append(line)
            continue
        if stripped and not any(p in stripped for p in ("Fix", "Debug", "What", "Why", "How", "//", "#")):
            if lines.index(line) > 0 and code_lines:
                prev = code_lines[-1].strip()
                if prev.endswith(":") or prev.endswith("\\") or prev.endswith(","):
                    code_lines.append(line)
    return "\n".join(code_lines)


def solve(prompt: str) -> tuple[Optional[str], Optional[float]]:
    lower = prompt.lower()

    contains_code = "def " in prompt or "class " in prompt or "return " in prompt
    if contains_code:
        code = _extract_code(prompt)
        if not code.strip():
            code = prompt

        result = _detect_mutable_defaults(code)
        if result:
            return f"Bug found: {result}", 0.90

        result = _detect_missing_return(code)
        if result:
            return f"Bug found: {result}", 0.85

        result = _detect_off_by_one(code)
        if result:
            return f"Bug found: {result}", 0.85

        result = _detect_recursion_base(code)
        if result:
            return f"Bug found: {result}", 0.80

        result = _detect_assignment_in_condition(code)
        if result:
            return f"Bug found: {result}", 0.90

        result = _detect_missing_colon(code)
        if result:
            return f"Bug found: {result}", 0.85

        return None, None

    for name, keywords in TEMPLATE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return TEMPLATES[name], 0.90

    return None, None
