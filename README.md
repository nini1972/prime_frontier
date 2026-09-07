# Prime Frontier

An empirical number-theory laboratory that combines classical computational
prime-number engines with an autonomous, LLM-driven "Agent World" that
formulates, tests, and archives novel mathematical conjectures about primes
— then compiles everything into an interactive dashboard.

## What's in here

```
core/                 Deterministic math engines (no AI, no network)
  sieve.py              Segmented prime sieve, gap statistics, Cramer-conjecture records
  modular_orbits.py      Lemke Oliver-Soundararajan consecutive-prime modular bias (mod 10 / mod 30)
  zeta_harmonics.py       Riemann zeta zero harmonics & the explicit-formula prime staircase
  goldbach.py             Goldbach partition counts g(n) and record-low tracking
  domains.py              Registry that turns each engine above into a "research domain"
                          the AI agents can be pointed at

models/                The autonomous research agents
  llm_client.py           Zero-dependency OpenRouter client (DeepSeek R1, Qwen 2.5 Math, Claude)
  math_researcher.py      Runs one research cycle: harvest data -> ask the LLM for a
                          conjecture -> execute its verification script -> self-repair
                          on errors -> log the verdict
  knowledge_base.py       Persistent memory: per-domain coverage stats, duplicate-conjecture
                          detection, confirmed-law "hall of fame"
  agent_world.py          Orchestrator that runs many research cycles in a loop, rotating
                          domains and models

visualizer/            Presentation layer
  compile_dashboard.py    Compiles all of the above into a single self-contained dashboard.html
  ulam_sacks.py            Generates the Ulam / Sacks prime spiral images

outputs/               Generated artifacts (plots, conjectures.json, knowledge_base.json)
dashboard.html          The compiled interactive dashboard (open directly in a browser)
```

## Requirements

- Python 3.9+
- The packages in [requirements.txt](requirements.txt): `numpy`, `matplotlib`, `mpmath`
- (Optional, only needed for the AI agents) An [OpenRouter](https://openrouter.ai/) API key

## Setup

```powershell
# From the repository root
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

### 1. Generate the dashboard (no API key needed)

This runs the sieve, modular-bias, and zeta-harmonics engines over real data and
compiles everything already in `outputs/` (including any past AI conjectures)
into `dashboard.html`:

```powershell
python -m visualizer.compile_dashboard
```

Then open `dashboard.html` in a browser (double-click it, or `start dashboard.html`
on Windows).

### 2. (Optional) Enable the AI research agents

The agents need an OpenRouter API key. Provide it either as an environment
variable:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
```

or in a `config\.env` file at the repo root (already gitignored):

```
OPENROUTER_API_KEY=sk-or-...
```

Without a key, the agent scripts still run and print a clear "DRY RUN" notice
instead of crashing — useful for sanity-checking the pipeline.

### 3. Run a single research cycle

```powershell
python -m models.math_researcher
```

Picks one research domain (coverage-weighted), asks the model for a novel
conjecture, executes its self-contained verification script (self-repairing
once or twice if the script errors out), and appends the result to
`outputs/conjectures.json`.

### 4. Run the autonomous Agent World (multiple cycles)

```powershell
python -m models.agent_world --cycles 10 --sleep 5 --models haiku,sonnet
```

Common options:

| Flag            | Meaning                                                              | Default |
|-----------------|-----------------------------------------------------------------------|---------|
| `--cycles`      | Number of research cycles to run                                     | `3`     |
| `--sleep`       | Seconds to wait between cycles (be kind to the API)                  | `5.0`   |
| `--models`      | Comma-separated OpenRouter model aliases to rotate through            | `haiku` |
| `--domain`      | Pin every cycle to one domain instead of letting the explorer choose  | (auto)  |
| `--max-repairs` | Max self-repair attempts per broken verification script               | `2`     |

Model aliases (see [models/llm_client.py](models/llm_client.py)):
`haiku`, `sonnet`, `r1`, `r1_full`, `qwen_math`.

### 5. Re-compile the dashboard to see new results

```powershell
python -m visualizer.compile_dashboard
```

Re-run this any time after step 3/4 to refresh `dashboard.html` with the
latest conjectures and Agent World coverage stats.

## Running the core engines standalone

Each engine in `core/` has a self-test you can run directly, useful for
verifying the math independent of any AI/network dependency:

```powershell
python -m core.sieve             # prime sieve + gap statistics
python -m core.modular_orbits     # modular transition bias + heatmap plot
python -m core.zeta_harmonics     # zeta zero harmonics + staircase plot
python -m core.goldbach           # Goldbach partition counts
python -m core.domains            # sanity-check every research domain's data + prompt
python -m models.knowledge_base   # sanity-check the knowledge base / dedup logic
```

## Reproducibility

- **`core/*` and the dashboard's numeric content are deterministic** — same
  inputs always produce the same statistics and plots.
- **The AI research cycles are intentionally non-deterministic.** Conjectures
  are sampled from an LLM (temperature > 0) and the whole point is to keep
  discovering *new* ideas, so re-running `math_researcher.py` / `agent_world.py`
  will generally produce different conjectures, ids, and timestamps each time.
- `outputs/knowledge_base.json` is what keeps runs from repeating themselves:
  it remembers what's already been tried so the world explores new territory
  and skips near-duplicate conjectures instead of re-discovering the same one.

## Outputs

| File                                   | Produced by                     | Contents |
|-----------------------------------------|----------------------------------|----------|
| `outputs/conjectures.json`              | `math_researcher.py` / `agent_world.py` | Every AI-generated conjecture, its verification code, and verdict |
| `outputs/knowledge_base.json`           | same                              | Agent World coverage stats, confirmed laws, dedup memory |
| `outputs/*.png`                         | `core/*.py` / `visualizer/ulam_sacks.py` | Modular bias heatmap, zeta staircase, Ulam/Sacks spirals |
| `dashboard.html`                        | `visualizer/compile_dashboard.py` | The interactive dashboard (open in a browser) |
