from app.deterministic import math_solver


class TestMathSolver:
    def test_addition(self):
        ans, conf = math_solver.solve("What is 18 + 24?")
        assert ans == "42"
        assert conf and conf >= 0.9

    def test_subtraction(self):
        ans, conf = math_solver.solve("Calculate 100 - 37.")
        assert ans == "63"
        assert conf and conf >= 0.9

    def test_percentage(self):
        ans, conf = math_solver.solve("What is 15% of 200?")
        assert ans == "30.0"
        assert conf and conf >= 0.9

    def test_average(self):
        ans, conf = math_solver.solve("Average of 10, 20, 30.")
        assert ans == "20"
        assert conf and conf >= 0.9

    def test_solve_equation(self):
        ans, conf = math_solver.solve("Solve x + 5 = 12.")
        assert ans == "7"
        assert conf and conf >= 0.9

    def test_multiplication(self):
        ans, conf = math_solver.solve("What is 7 × 8?")
        assert ans == "56"
        assert conf and conf >= 0.9

    def test_no_match(self):
        assert math_solver.solve("Write a poem") == (None, None)

    def test_empty(self):
        assert math_solver.solve("") == (None, None)
