"""Tests for vector problems (parsing, solving, pipeline integration)."""

from __future__ import annotations

import pytest
import sympy
from sympy import Matrix

from services.math_service import MathService, MathServiceError
from src.latex_parser import LatexParser
from src.vector_ops import VectorError, VectorParser, VectorProblem, solve_vector


@pytest.fixture(scope="module")
def parser() -> VectorParser:
    return VectorParser(LatexParser()._latex_to_sympy)


@pytest.fixture(scope="module")
def service() -> MathService:
    return MathService()


class TestParsing:
    @pytest.mark.parametrize(("latex", "op"), [
        (r"(1, 2, 3) \cdot (4, 5, 6)", "dot"),
        (r"\langle 1, -1 \rangle \times \langle 2, 3 \rangle", "cross"),
        (r"(1, 2, 3) + (4, 5, 6)", "add"),
        (r"(5, 7) - (2, 3)", "subtract"),
        (r"| ( 3 , 4 ) |", "magnitude"),
        (r"(2i + 3j - k) \cdot (i - j + 2k)", "dot"),
        (r"\begin {pmatrix} 1 \\ 2 \\ 2 \end {pmatrix} \cdot \begin {pmatrix} 3 \\ 0 \\ 4 \end {pmatrix}", "dot"),
    ])
    def test_notations_and_operations(self, parser: VectorParser, latex: str, op: str) -> None:
        problem = parser.parse(latex)
        assert problem is not None and problem.operation == op

    def test_named_vectors_with_definitions(self, parser: VectorParser) -> None:
        problem = parser.parse(
            r"\text{Find } \vec{a} \cdot \vec{b} \text{ for } \vec{a}=(2,-1,3), \vec{b}=(-2,-2,2)")
        assert problem.vectors[0].tolist() == [[2], [-1], [3]]
        assert problem.labels == ["\\vec{a}", "\\vec{b}"]

    def test_hat_notation_with_where_clause(self, parser: VectorParser) -> None:
        problem = parser.parse(r"\vec{a} \times \vec{b} \text{ where } \vec{a} = 2\hat{i} + 3\hat{j} - \hat{k}, "
                               r"\vec{b} = \hat{i} - \hat{j} + 2\hat{k}")
        assert problem.vectors[0].tolist() == [[2], [3], [-1]]

    def test_inline_definition(self, parser: VectorParser) -> None:
        problem = parser.parse(r"\text{Find the magnitude of } \vec{a} = 2\hat{i} - 2\hat{j} + \hat{k}")
        assert problem.operation == "magnitude" and problem.vectors[0].tolist() == [[2], [-2], [1]]

    def test_angle_between(self, parser: VectorParser) -> None:
        problem = parser.parse(r"\text{Find the angle between } \vec{a} \text{ and } \vec{b} "
                               r"\text{ for } \vec{a}=(1,0,0), \vec{b}=(1,1,0)")
        assert problem.operation == "angle"

    @pytest.mark.parametrize("latex", [
        r"x^2 - 5x + 6 = 0", r"\sqrt{49}", r"y = 2x + 3", r"2 x \cdot 3", r"(x+1)(x-1)",
        r"\begin {pmatrix} 3 & 4 \\ 1 & 2 \end {pmatrix} \cdot \begin {pmatrix} 1 & 0 \\ 0 & 1 \end {pmatrix}",
        r"\text{Find the distance between } (1, 9) \text{ and } (5, 10)",
    ])
    def test_non_vector_input_is_left_alone(self, parser: VectorParser, latex: str) -> None:
        assert parser.parse(latex) is None


class TestSolving:
    @staticmethod
    def _solve(op: str, *vectors: list) -> sympy.Basic:
        problem = VectorProblem(op, [Matrix(v) for v in vectors], ["\\vec{a}", "\\vec{b}"][: len(vectors)])
        return solve_vector(problem)

    def test_dot_product(self) -> None:
        assert self._solve("dot", [1, 2, 3], [4, 5, 6]).value == 32

    def test_dot_product_perpendicular_remark(self) -> None:
        solution = self._solve("dot", [1, 0], [0, 5])
        assert solution.value == 0 and "perpendicular" in " ".join(s[1] for s in solution.steps)

    def test_cross_product_known_value(self) -> None:
        solution = self._solve("cross", [2, -1, 3], [1, 4, -2])
        assert solution.value.tolist() == [[-10], [7], [9]]
        assert solution.details["area"] == sympy.sqrt(230)

    def test_cross_product_is_perpendicular_to_both(self) -> None:
        a, b = Matrix([3, -2, 5]), Matrix([1, 4, -1])
        cross = self._solve("cross", list(a), list(b)).value
        assert cross.dot(a) == 0 and cross.dot(b) == 0

    def test_cross_product_2d_is_z_component(self) -> None:
        assert self._solve("cross", [1, -1], [2, 3]).value == 5

    def test_cross_product_needs_3_components(self) -> None:
        with pytest.raises(VectorError):
            self._solve("cross", [1, 2, 3, 4], [1, 2, 3, 4])

    def test_magnitude_and_unit_vector(self) -> None:
        solution = self._solve("magnitude", [3, 4])
        assert solution.value == 5
        assert solution.details["unit_vector"].tolist() == [[sympy.Rational(3, 5)], [sympy.Rational(4, 5)]]

    def test_zero_vector_has_no_unit_vector(self) -> None:
        assert "unit_vector" not in self._solve("magnitude", [0, 0, 0]).details

    def test_sum_and_difference(self) -> None:
        assert self._solve("add", [1, 2, 3], [4, 5, 6]).value.tolist() == [[5], [7], [9]]
        assert self._solve("subtract", [5, 7], [2, 3]).value.tolist() == [[3], [4]]

    def test_angle_45_degrees(self) -> None:
        solution = self._solve("angle", [1, 0, 0], [1, 1, 0])
        assert solution.value == sympy.pi / 4 and solution.numeric[0] == pytest.approx(45.0)

    def test_angle_with_zero_vector_is_an_error(self) -> None:
        with pytest.raises(VectorError):
            self._solve("angle", [0, 0, 0], [1, 2, 3])

    def test_dimension_mismatch(self) -> None:
        with pytest.raises(VectorError):
            self._solve("dot", [1, 2], [1, 2, 3])


class TestPipeline:
    @pytest.mark.parametrize(("latex", "answer_part"), [
        (r"(1, 2, 3) \cdot (4, 5, 6)", "= 32"),
        (r"(2, -1, 3) \times (1, 4, -2)", "-10\\hat{i} + 7\\hat{j} + 9\\hat{k}"),
        (r"| ( 3 , 4 ) |", "= 5"),
        (r"\text{Find the angle between } \vec{a} \text{ and } \vec{b} \text{ for } \vec{a}=(1,0,0), \vec{b}=(1,1,0)", "45.00"),
    ])
    def test_solved_with_steps_and_no_graph(self, service: MathService, latex: str, answer_part: str) -> None:
        result = service.solve_latex(latex)
        assert result["problem_type"] == "vector"
        assert answer_part in result["answer"]
        assert len(result["steps"]) >= 4 and result["graph_base64"] is None

    def test_mismatched_dimensions_is_a_friendly_error(self, service: MathService) -> None:
        with pytest.raises(MathServiceError):
            service.solve_latex(r"(1, 2) \cdot (1, 2, 3)")

    def test_matrix_product_is_still_a_matrix_problem(self, service: MathService) -> None:
        latex = r"\begin {pmatrix} 3 & 4 \\ 1 & 2 \end {pmatrix} \begin {pmatrix} 1 & 0 \\ 0 & 1 \end {pmatrix}"
        assert service.solve_latex(latex)["problem_type"] == "matrix"

    def test_vector_practice_problems_round_trip(self, service: MathService) -> None:
        from src.problem_generator import ProblemGenerator

        generator = ProblemGenerator(seed=11)
        for _ in range(10):
            problem = generator.generate_similar("vectors")
            assert service.solve_latex(problem.latex_problem)["problem_type"] == "vector"
