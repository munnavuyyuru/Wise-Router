from app.deterministic import logic_solver


class TestLogicSolver:
    def test_tallest(self):
        ans, conf = logic_solver.solve("Alice is taller than Bob. Bob is taller than Charlie. Who is tallest?")
        assert ans == "alice"
        assert conf and conf >= 0.8

    def test_slowest(self):
        ans, conf = logic_solver.solve("Diana is faster than Eve. Eve is faster than Frank. Who is slowest?")
        assert ans in ("frank", "frank is slowest")
        assert conf and conf >= 0.8
