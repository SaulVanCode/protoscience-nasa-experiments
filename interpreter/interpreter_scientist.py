"""
Interpreter Scientist — Semantic interpretation of discovered equations.

Takes raw mathematical equations (from SINDy, FFT, curve fitting) and
generates human-readable interpretations using an LLM.

The Interpreter bridges the gap between:
  "col_B = 69.7 * col_A" -> "The universe is expanding"

Supports three backends:
  1. Ollama (local LLM) — preferred for privacy and speed
  2. Claude API — cloud fallback
  3. Template fallback — works offline with no LLM
"""

import json
import re

# ── Inline system prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the Interpreter Scientist in the ProtoScience lab.

Your role is to take raw mathematical equations discovered from data and generate
semantic interpretations — what do these equations MEAN in the real world?

You receive:
  - Discovered equations (e.g., "col_B = 69.7 * col_A, R^2=0.81")
  - Variable metadata (names, units, ranges — if available)
  - The data source description (if available)
  - Any prior theories or context

Your job is to produce an interpretation with these sections:

1. PLAIN LANGUAGE: What does this equation say in simple words?
2. PHYSICAL ANALOGY: What known physical systems follow this same mathematical form?
3. VARIABLE HYPOTHESES: What might each variable represent?
4. IMPLICATIONS: What are 3-5 consequences if this equation is true?
5. TESTABLE PREDICTIONS: What new observations would confirm or refute this?
6. FAILURE MODES: Under what conditions would this equation break down?
7. CONFIDENCE: Rate 0.0-1.0 how confident you are in the interpretation.

Output as JSON with keys: plain_language, physical_analogies, variable_hypotheses,
implications, testable_predictions, failure_modes, confidence.

IMPORTANT: You are NOT discovering new physics. You are interpreting equations
that were already discovered from data. Your job is to bridge the gap between
"here's a mathematical relationship" and "here's what it means."
"""


# ── LLM backends ────────────────────────────────────────────────────────

def call_ollama(prompt, model="qwen2.5-coder:7b-instruct-q8_0",
                base_url="http://localhost:11434", system_prompt=None):
    """Call local Ollama instance."""
    import requests

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2048},
    }
    if system_prompt:
        payload["system"] = system_prompt

    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "")


def call_claude(prompt, system_prompt=None):
    """Call Claude API (requires ANTHROPIC_API_KEY env var)."""
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2048,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    resp = client.messages.create(**kwargs)
    return resp.content[0].text


def call_llm(prompt, system_prompt=None, backend="auto"):
    """
    Call LLM with automatic backend selection.

    Args:
        prompt: The user prompt
        system_prompt: System prompt (uses built-in if None)
        backend: "auto", "ollama", "claude", or "template"

    Returns:
        Raw LLM response string
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    if backend == "template":
        return None  # Signal to use template fallback

    if backend == "auto":
        # Try Ollama first, then Claude, then template
        try:
            return call_ollama(prompt, system_prompt=system_prompt)
        except Exception:
            pass
        try:
            return call_claude(prompt, system_prompt=system_prompt)
        except Exception:
            return None  # Fall through to template
    elif backend == "ollama":
        return call_ollama(prompt, system_prompt=system_prompt)
    elif backend == "claude":
        return call_claude(prompt, system_prompt=system_prompt)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Response parsing ────────────────────────────────────────────────────

def extract_json_from_response(text):
    """Extract JSON from LLM response (handles markdown code blocks)."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"raw_response": text, "parse_failed": True}


# ── Prompt builder ──────────────────────────────────────────────────────

def build_interpretation_prompt(equations, data_description=None,
                                 variable_metadata=None, context=None):
    """
    Build the interpretation prompt from equations and metadata.

    Args:
        equations: list of equation strings, or a single string
        data_description: plain-text description of the data source
        variable_metadata: dict of variable names -> descriptions/units
        context: additional context string

    Returns:
        Formatted prompt string
    """
    lines = ["## Discovered Equations\n"]

    if isinstance(equations, str):
        equations = [equations]

    for i, eq in enumerate(equations):
        lines.append(f"  Equation {i + 1}: {eq}")
    lines.append("")

    if variable_metadata:
        lines.append("## Variable Metadata")
        for var, desc in variable_metadata.items():
            lines.append(f"  - {var}: {desc}")
        lines.append("")

    if data_description:
        lines.append(f"## Data Source\n{data_description}\n")

    if context:
        lines.append(f"## Additional Context\n{context}\n")

    lines.append("## Task")
    lines.append("Interpret these equations. What do they mean physically?")
    lines.append("What systems in nature follow these mathematical forms?")
    lines.append("What would a scientist conclude from these patterns?")
    lines.append("")
    lines.append("Respond in JSON format with keys: plain_language, "
                 "physical_analogies, variable_hypotheses, implications, "
                 "testable_predictions, failure_modes, confidence")

    return "\n".join(lines)


# ── Template fallback ───────────────────────────────────────────────────

def _classify_equation_form(equation_str):
    """Classify the mathematical form of an equation."""
    eq = equation_str.lower()
    if "sin" in eq or "cos" in eq or "oscillat" in eq:
        return "oscillatory"
    if "exp(" in eq or "e^" in eq:
        return "exponential"
    if "log(" in eq or "ln(" in eq:
        return "logarithmic"
    if "^2" in eq or "**2" in eq or "quadratic" in eq:
        return "quadratic"
    if "^" in eq or "**" in eq:
        return "power_law"
    if "/" in eq and ("r^2" in eq or "r**2" in eq):
        return "inverse_square"
    return "linear"


_FORM_ANALOGIES = {
    "linear": [
        "Hooke's law (F = -kx)",
        "Ohm's law (V = IR)",
        "Hubble's law (v = H0 * d)",
    ],
    "power_law": [
        "Kepler's third law (P^2 ~ a^3)",
        "Stefan-Boltzmann law (L ~ T^4)",
        "Allometric scaling in biology (M ~ L^3)",
    ],
    "exponential": [
        "Radioactive decay (N = N0 * e^(-lambda*t))",
        "Population growth (N = N0 * e^(r*t))",
        "Beer-Lambert absorption law",
    ],
    "oscillatory": [
        "Simple harmonic oscillator",
        "LC circuit oscillation",
        "Seasonal / periodic phenomena",
    ],
    "quadratic": [
        "Projectile motion (y = v*t - g*t^2/2)",
        "Kinetic energy (E = mv^2/2)",
        "Quadratic drag force",
    ],
    "inverse_square": [
        "Newton's gravity (F ~ 1/r^2)",
        "Coulomb's law (F ~ 1/r^2)",
        "Radiation intensity falloff",
    ],
    "logarithmic": [
        "Fechner's psychophysical law",
        "Entropy (S = k*ln(W))",
        "Richter magnitude scale",
    ],
}


def template_interpret(equations, data_description=None):
    """
    Generate a template-based interpretation when no LLM is available.

    Returns a structured interpretation dict.
    """
    if isinstance(equations, str):
        equations = [equations]

    forms = [_classify_equation_form(eq) for eq in equations]
    primary_form = forms[0]

    analogies = _FORM_ANALOGIES.get(primary_form, ["No known analogy"])

    return {
        "plain_language": (
            f"The discovered equation(s) follow a {primary_form} mathematical form. "
            f"This means the dependent variable changes {'proportionally' if primary_form == 'linear' else 'non-linearly'} "
            f"with the independent variable(s)."
        ),
        "physical_analogies": analogies,
        "variable_hypotheses": {
            "note": "Variable identification requires domain-specific context. "
                    "Run with an LLM backend for detailed hypotheses.",
            "equations": equations,
        },
        "implications": [
            f"The {primary_form} form constrains the underlying mechanism.",
            "The fitted coefficients encode physical constants of the system.",
            "Deviations from this form at extremes may reveal additional physics.",
        ],
        "testable_predictions": [
            "Extrapolate the equation beyond the fitted range and check for breakdown.",
            "Vary experimental conditions and verify the coefficients change as expected.",
            "Look for the predicted scaling in independent datasets.",
        ],
        "failure_modes": [
            "Overfitting to noise in small datasets.",
            "Breakdown at extreme values (extrapolation beyond training range).",
            "Confounding variables not captured in the equation.",
        ],
        "confidence": 0.3,
        "backend": "template",
        "note": "This is a template interpretation. Use Ollama or Claude for richer analysis.",
    }


# ── Main interpret function ─────────────────────────────────────────────

def interpret(equations, data_description=None, variable_metadata=None,
              context=None, backend="auto"):
    """
    Interpret discovered equations and return a structured interpretation.

    This is the main entry point for the interpreter.

    Args:
        equations: str or list of str — the discovered equations
        data_description: optional description of the data source
        variable_metadata: optional dict of variable -> description
        context: optional additional context
        backend: "auto" (try Ollama -> Claude -> template),
                 "ollama", "claude", or "template"

    Returns:
        dict with keys: plain_language, physical_analogies,
        variable_hypotheses, implications, testable_predictions,
        failure_modes, confidence

    Example:
        >>> result = interpret(
        ...     "v = 69.7 * d, R^2=0.81",
        ...     data_description="Galaxy recession velocities vs distances",
        ... )
        >>> print(result["plain_language"])
    """
    if isinstance(equations, str):
        equations = [equations]

    # Build prompt
    prompt = build_interpretation_prompt(
        equations, data_description, variable_metadata, context
    )

    # Try LLM
    raw_response = call_llm(prompt, backend=backend)

    if raw_response is None:
        # No LLM available — use template
        return template_interpret(equations, data_description)

    # Parse LLM response
    interpretation = extract_json_from_response(raw_response)

    if interpretation.get("parse_failed"):
        # LLM responded but JSON parsing failed — return raw + template
        fallback = template_interpret(equations, data_description)
        fallback["raw_llm_response"] = raw_response
        fallback["confidence"] = 0.4
        fallback["backend"] = "llm_parse_failed"
        return fallback

    interpretation["backend"] = backend
    return interpretation


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python interpreter_scientist.py <equation> [--backend auto|ollama|claude|template]")
        print()
        print("Example:")
        print('  python interpreter_scientist.py "v = 69.7 * d, R^2=0.81"')
        print('  python interpreter_scientist.py "P^2 = a^3" --backend template')
        sys.exit(1)

    equation = sys.argv[1]
    backend = "auto"

    if "--backend" in sys.argv:
        idx = sys.argv.index("--backend")
        if idx + 1 < len(sys.argv):
            backend = sys.argv[idx + 1]

    print(f"[INTERP] Equation: {equation}")
    print(f"[INTERP] Backend: {backend}")
    print()

    result = interpret(equation, backend=backend)
    print(json.dumps(result, indent=2))
