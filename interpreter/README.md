# Interpreter Scientist

Semantic interpretation of discovered mathematical equations. Takes raw equations (from SINDy, FFT, curve fitting, etc.) and generates human-readable interpretations explaining what they mean physically.

## Quick Start

```python
from interpreter_scientist import interpret

result = interpret(
    "v = 69.7 * d, R^2=0.81",
    data_description="Galaxy recession velocities vs distances from NED-D catalog",
)

print(result["plain_language"])
# "Variable v increases proportionally with d at rate 69.7..."

print(result["physical_analogies"])
# ["Hooke's law (F = -kx)", "Ohm's law (V = IR)", "Hubble's law (v = H0 * d)"]
```

## Backends

The interpreter supports three LLM backends, tried in order when `backend="auto"`:

| Backend | Requires | Quality | Speed |
|---------|----------|---------|-------|
| `ollama` | Local Ollama server running | High | Fast |
| `claude` | `ANTHROPIC_API_KEY` env var | Highest | Network |
| `template` | Nothing | Basic | Instant |

### Using Ollama (recommended for local use)

```bash
# Install and start Ollama
ollama pull qwen2.5-coder:7b-instruct-q8_0
ollama serve

# Run interpreter
python interpreter_scientist.py "P^2 = a^3"
```

### Using Claude API

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python interpreter_scientist.py "P^2 = a^3" --backend claude
```

### Template fallback (no LLM needed)

```bash
python interpreter_scientist.py "P^2 = a^3" --backend template
```

## API

### `interpret(equations, data_description=None, variable_metadata=None, context=None, backend="auto")`

Main entry point. Returns a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `plain_language` | str | Simple explanation of the equation |
| `physical_analogies` | list[str] | Known physical systems with the same form |
| `variable_hypotheses` | dict | Possible interpretations for each variable |
| `implications` | list[str] | Consequences if the equation is true |
| `testable_predictions` | list[str] | Falsifiable predictions |
| `failure_modes` | list[str] | When the equation might break down |
| `confidence` | float | 0.0-1.0 confidence in the interpretation |

### Parameters

- **equations**: `str` or `list[str]` -- the discovered equation(s)
- **data_description**: optional context about the data source
- **variable_metadata**: optional `dict` mapping variable names to descriptions
- **context**: optional additional context string
- **backend**: `"auto"`, `"ollama"`, `"claude"`, or `"template"`

## CLI Usage

```bash
python interpreter_scientist.py "<equation>" [--backend auto|ollama|claude|template]
```

## Dependencies

- **Core**: Python 3.8+ (no external deps for template mode)
- **Ollama backend**: `requests`
- **Claude backend**: `anthropic`

## Origin

Simplified from the [proto-science-organism](https://github.com/SaulVanCode/proto-science-organism) multi-agent scientific discovery system. The original version includes ledger event tracking and artifact file management for the full discovery pipeline.
