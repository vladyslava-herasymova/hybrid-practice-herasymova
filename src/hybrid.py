"""HYBRID (neuro-symbolic) pipeline:

    problem text --(LLM formalizer)--> IR --(validation)--> Z3 solver --(presentation)--> answer

This is the main thing your team builds. The plumbing is here; the parts marked TODO(team) are yours.
Suggested order of work:
    1. write FORMALIZER_PROMPT and get formalize() returning sensible IR for one simple problem;
    2. implement validate_ir();
    3. add ONE retry with feedback in solve();
    4. fix run_solver() so that "unique" and "multiple" can be told apart;
    5. (extension) replace the LLM-based present() with deterministic code for some problem types.

See docs/ir.md for the IR format.
"""
import json
from typing import Any, Optional

from pydantic import BaseModel, Field

import ir_solver
import llm
from answer import Answer


class HybridError(Exception):
    """The hybrid pipeline could not produce an answer (recorded by run.py as an error, not as status "none")."""


class IRModel(BaseModel):
    """JSON schema the formalizer must follow. Keep it in sync with docs/ir.md."""
    variables: dict[str, Any]
    all_different: list[list[str]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objective: Optional[dict[str, str]] = None


# Compact IR description for the model. You may edit it, but keep it consistent with ir_solver.py.
IR_SPEC = """IR format (JSON):
{
  "variables": {"<name>": [lo, hi]  or  {"values": [v1, v2, ...]}},     integer variables and their domains
  "all_different": [["<name>", ...], ...],                                 groups whose values must all differ
  "constraints": ["<expression>", ...],                                    every expression must be true
  "objective": {"maximize": "<expression>"}  or  {"minimize": "<expression>"}   ONLY for optimisation problems
}

Variable names must be identifiers: letters, digits and underscores only, no spaces.
Variable values must be ranges of integers. Variable value must be an array.
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
The maximum number of constraints allowed is 20."""


SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."

# TODO(team): this prompt is the heart of the hybrid pipeline. Things worth considering:
#   * how should the model choose variables? (one variable per entity? per attribute value?)
#   * how should it encode positions, days, "morning/afternoon", "next to", "between", "at most"?
#   * facts that the text does not state but that the model must know (calendar, working week, ...);
#   * a worked example (few-shot) on a puzzle that is NOT in your set;
#   * how to encode knights and knaves: a statement S by person X becomes  X == (S).

FORMALIZER_PROMPT = """Translate the puzzle below into the IR format. Do not solve the puzzle.

If any name consists of more than one word, you need to use an underscore to connect them.
Example: "Morning run" should be translated to "Morning_run".


Full example:
Lisa, Mark and Carla are destributing red, blue and green pens between themselves. Mark doesn't want the blue pen. Carla doesn't want the red pen and the blue pen.

Resulting json:
{{
  "variables": {{"Lisa": [1, 3], "Mark": [1, 3], "Carla": [1, 3]}},
  "all_different": [["Lisa", "Mark", "Carla"]],
  "constraints": ["Mark != 2", "(Carla != 1) and (Carla != 2)"]
}}



Full example2:
Three trains leave from platforms 1, 2 and 3: an intercity, a regional and an express, one per platform. The express does not leave from platform 1. The regional leaves from a platform with a lower number than the intercity. The intercity does not leave from platform 3.

Resulting json:
{{
  "variables": {{"Intercity": [1, 3], "Regional": [1, 3], "Express": [1, 3]}},
  "all_different": [["Intercity", "Regional", "Express"]],
  "constraints": ["Express != 1", "Regional < Intercity", "Intercity != 3"]
}}


{ir_spec}

Puzzle:
{text}
{feedback}"""

FORMALIZER_PROMPT2 = """Translate the puzzle below into the IR format. Do not solve the puzzle.

If letters stand for digits in a number, such numbers should be called CompositeNumbers. An example of a CompositeNumber: ABCD.
To get a normal integer number each letter in the CompositeNumber should be replaced by the corresponding digit.

Parts of the day should be treated as separate values and follow a sequential order.
Parts of the day should be ordered by day of the week and then by the part of the day.
If each day has N slots and there are M days, then there are N * M slots in total and the values range from 1 to N * M.
The X time of the Y day is represented by an integer value that equals (X - 1) * N + Y, where N is the total amount of times of the day mentioned in the puzzle.

"in front of" means "less than" if the order is from front (1) to back (n) and "more than" if the order is from back (1) to front (n).
Example: Mark in in the back and Tom is in front of Mark. The order is from front (1) to back (2).
Resulting constraint: "Tom < Mark"

If any name consists of more than one word, you need to use an underscore to connect them.
Example: "Morning run" should be translated to "Morning_run".

The youngest means smaller than any other entities mentioned. It can still be any number in [lo, hi].
Example: Tom, Jeremy and Lisa are siblings. Lisa is the youngest.
Resulting constraints: ["Lisa < Tom", "Lisa < Jeremy"].

Full example:
Three trains leave from platforms 1, 2 and 3: an intercity, a regional and an express, one per platform. The express does not leave from platform 1. The regional leaves from a platform with a lower number than the intercity. The intercity does not leave from platform 3.

Resulting json:
{{
  "variables": {{"Intercity": [1, 3], "Regional": [1, 3], "Express": [1, 3]}},
  "all_different": [["Intercity", "Regional", "Express"]],
  "constraints": ["Express != 1", "Regional < Intercity", "Intercity != 3"]
}}


{ir_spec}

Puzzle:
{text}
{feedback}"""

FORMALIZER_PROMPT3 = """Translate the puzzle below into the IR format. Do not solve the puzzle.

If letters stand for digits in a number, such numbers should be called CompositeNumbers. An example of a CompositeNumber: ABCD.
CompositeNumber consisting of 4 letters-digits can be calculated as follows: ABCD == A * 1000 + B * 100 + C * 10 + D.
CompositeNumber consisting of 5 letters-digits can be calculated as follows: ABCDE == A * 10000 + B * 1000 + C * 100 + D * 10 + E.
Other CompositeNumbers should be calculated in a similar manner.
If the CompositeNumber is even, than the last letter of this CompositeNumber represents an even digit (0, 2, 4, 6 or 8).
An expression of type PF + GREQ ----- FRON should be transformed to an expression of type PF + GREQ == FRON. Here PF, GREQ and FRON are CompositeNumbers.
CompositeNumbers should not be used as variables. They should be replaced by the calculations using the letters that are used as variables instead.


Morning slot means morning part of the day.
Afternoon slot means afternoon part of the day.
Evening slot means evening part of the day.
Night slot means night part of the day.
Parts of the day should be treated as separate values and follow a sequential order.
Parts of the day should be ordered by day of the week and then by the part of the day.
If each day has N slots and there are M days, then there are N * M slots in total and the values range from 1 to N * M.
The X slot of the Y day is represented by an integer value that equals (X - 1) * N + Y, where N is the total amount of slots mentioned in the puzzle.

"in front of" means "less than" if the order is from front (1) to back (n) and "more than" if the order is from back (1) to front (n).
Example: Mark in in the back and Tom is in front of Mark. The order is from front (1) to back (2).
Resulting constraint: "Tom < Mark"

If any name consists of more than one word, you need to use an underscore to connect them.
Example: "Morning run" should be translated to "Morning_run".

The youngest means smaller than any other entities mentioned. It can still be any number in [lo, hi].
Example: Tom, Jeremy and Lisa are siblings. Lisa is the youngest.
Resulting constraints: ["Lisa < Tom", "Lisa < Jeremy"].

Full example:
Three trains leave from platforms 1, 2 and 3: an intercity, a regional and an express, one per platform. The express does not leave from platform 1. The regional leaves from a platform with a lower number than the intercity. The intercity does not leave from platform 3.

Resulting json:
{{
  "variables": {{"Intercity": [1, 3], "Regional": [1, 3], "Express": [1, 3]}},
  "all_different": [["Intercity", "Regional", "Express"]],
  "constraints": ["Express != 1", "Regional < Intercity", "Intercity != 3"]
}}


{ir_spec}

Puzzle:
{text}
{feedback}"""


def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    fb = f"\nYour previous encoding was rejected:\n{feedback}\nFix these problems.\n" if feedback else ""
    print(fb)
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=problem["text"], feedback=fb)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM)
    return parsed.model_dump(exclude_none=True)


def validate_ir(ir: dict) -> list:
    """Return a list of human-readable error messages; an empty list means the IR is valid.

    TODO(team). Things to check:
      * every domain is [lo, hi] with lo <= hi, or {"values": [...]} with integers;
      * every name used in all_different / constraints / objective is a declared variable;
      * every constraint parses and uses only allowed syntax.
    Hint: ir_solver.build(ir) raises ValueError with a readable message for unknown variables and
    unsupported syntax - you can call it inside try/except and turn the exception into an error message.
    """
    try:
        for name, dom in ir["variables"].items():
            for d in dom:
                if not isinstance(d, int):
                    return ["Non-integer values found as variable values"]
        ir_solver.build(ir)
    except ValueError as e:
        print(f"Got build error: {e}")
        return [str(e)]
    return []


def run_solver(ir: dict) -> dict:
    """Run Z3 on the IR. Returns {"status": ..., "solutions": [...], "objective": ...}.

    TODO(team): with limit=1 the solver stops after the first solution, so it reports "unique" even when
    several solutions exist. Think about what limit you need to tell "unique" from "multiple".
    """
    return ir_solver.solve(ir, limit=5)


PRESENT_PROMPT = """A puzzle has been solved by a constraint solver. Rewrite the solver's result in the required
answer format. Do NOT solve the puzzle again - only translate the solver's assignment.

Puzzle:
{text}

Encoding used by the solver:
{ir}

Solver assignment (variable -> value):
{assignment}

{answer_format}"""


def present(problem: dict, ir: dict, result: dict) -> dict:
    """Turn the solver result into the required answer format.

    Provided baseline: the model translates the assignment. This is a weak point - the model can
    introduce errors AFTER the solver. Status and objective are always taken from the solver, never
    from the model. Extension: write deterministic code for problem types where the mapping is obvious.
    """
    status = result["status"]
    if status == "none":
        return Answer(status="none").model_dump()
    prompt = PRESENT_PROMPT.format(text=problem["text"], ir=json.dumps(ir, ensure_ascii=False),
                                   assignment=json.dumps(result["solutions"][0], ensure_ascii=False),
                                   answer_format=problem["answer_format"])
    parsed, raw = llm.ask_json(prompt, Answer)
    return Answer(status=status, solution=parsed.solution, objective=result.get("objective")).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    # TODO(team): if `errors` is not empty, make ONE more attempt:
    #   ir = formalize(problem, feedback="\n".join(errors)); errors = validate_ir(ir); record it in trace.
    if errors:
        print("Trying again with errors: " + "\n".join(errors))
        ir = formalize(problem, feedback="\n".join(errors))
        #raise HybridError("invalid IR: " + "; ".join(errors))

    try:
        result = run_solver(ir)
    except ValueError as e:                      # the solver rejected the IR
        raise HybridError(f"solver rejected the IR: {e}") from e
    trace["solver"] = {"status": result["status"], "n_solutions": len(result["solutions"]),
                       "objective": result.get("objective"),
                       #"solutions": result["solutions"]}
                       "first_solution": result["solutions"][0] if result["solutions"] else None}

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}

if __name__ == "main":
    print("hi")
