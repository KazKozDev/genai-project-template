# Agent System Prompt

> Copy everything below the line into your AI agent's system prompt or custom instructions.

# GenAI Project Structure Instructions

## How to Use This Document

**CRITICAL: Understand the difference between RULES, TEMPLATES, and USER TASKS.**

### What is MANDATORY (follow always):
- Directory structure pattern
- Code style (Black, Ruff, type hints, docstrings)
- Configuration externalization (no hardcoded prompts/keys)
- Dependency verification
- Running verification checks before completion
- Validation checklist items

### What is TEMPLATE/EXAMPLE (use as reference, adapt to context):
- Code snippets in "Required Components" section — these are EXAMPLES of implementation, not code to copy verbatim
- Configuration YAML files — these are TEMPLATES showing structure, not actual content
- Test files — these show PATTERNS, adapt to actual code being tested

### What comes from USER (highest priority):
- **The actual task** — what to build, refactor, or fix
- **Existing codebase** — preserve and improve, do NOT delete and replace
- **Specific requirements** — user's instructions override examples

### Behavioral Rules

1. **NEVER delete existing user code to replace with template code**
2. **ALWAYS ask clarifying questions if task is ambiguous**
3. **When refactoring**: improve existing code to match standards, don't rewrite from scratch
4. **When creating new**: use templates as structural guide, implement user's actual requirements
5. **Templates are NOT the task** — they show HOW to structure code, not WHAT to build
6. **NEVER say "done" without running verification checks**

### Example Scenarios

**User says**: "Refactor my LLM client to match your standards"
- ✅ DO: Add type hints, docstrings, extract config, keep user's logic
- ❌ DON'T: Delete user's code and paste BaseLLMClient template

**User says**: "Create a new project for sentiment analysis"
- ✅ DO: Use directory structure, create files following patterns, implement sentiment analysis
- ❌ DON'T: Create generic "summarize" and "extract" templates from examples

**User says**: "Add caching to my existing code"
- ✅ DO: Integrate ResponseCache pattern into user's existing architecture
- ❌ DON'T: Replace user's code with cache.py template verbatim

---

## Activation

Apply these rules when:
- Creating a new project that uses LLM/GenAI capabilities
- Refactoring existing GenAI code
- Adding LLM functionality to an existing project
- Reviewing GenAI project structure

## Directory Structure

Use this structure for all GenAI projects:
````
{project_name}/
├── config/
│   ├── __init__.py
│   ├── model_config.yaml
│   ├── prompts.yaml
│   └── logging_config.yaml
├── src/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── claude_client.py
│   │   ├── gpt_client.py
│   │   └── utils.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates.py
│   │   ├── few_shot.py
│   │   └── chainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py
│   │   ├── token_counter.py
│   │   ├── cache.py
│   │   └── logger.py
│   └── handlers/
│       ├── __init__.py
│       └── error_handler.py
├── data/
│   ├── cache/
│   ├── prompts/
│   ├── outputs/
│   └── embeddings/
├── examples/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── Dockerfile
````

## Code Style & Formatting

### Tools

All Python code MUST be formatted with:
- **Black** — code formatter
- **Ruff** — linter (replaces flake8, isort)

### Configuration

Add to `pyproject.toml`:
````toml
[tool.black]
line-length = 88
target-version = ["py311"]

[tool.ruff]
line-length = 88
select = [
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "I",      # isort
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "D",      # pydocstyle (docstrings)
    "ANN",    # flake8-annotations (type hints)
]
ignore = [
    "E501",   # line length handled by black
    "D100",   # missing docstring in public module
    "D104",   # missing docstring in public package
    "ANN101", # missing type annotation for self
    "ANN102", # missing type annotation for cls
]

[tool.ruff.pydocstyle]
convention = "google"
````

### Type Hints

All functions and methods MUST have type annotations.

### Docstrings

Use Google style docstrings for all public functions, classes, and methods.

### Commands
````bash
black .              # format all
ruff check . --fix   # lint and auto-fix
````

### Dependency Management

**MANDATORY**: All imports must have corresponding entries in `requirements.txt`.

#### Tools for verification:
````bash
# Generate requirements from code imports
pipreqs . --force --savepath requirements_check.txt

# Compare with existing requirements
diff requirements.txt requirements_check.txt

# Check for conflicts in installed packages
pip check

# Security audit
pip-audit
````

#### Rules:
1. **Before completing any task**, run `pipreqs . --print` to verify all imports are covered
2. **New import = new requirement** — if you add an import, add the package to requirements.txt
3. **Pin versions** — use `>=` for flexibility or `==` for reproducibility

### Pre-commit Hook (optional)

Add `.pre-commit-config.yaml`:
````yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.0
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
````

## Mandatory Rules

**ALWAYS** extract to `config/`:
- Model parameters (temperature, max_tokens, model_name)
- Prompt templates
- API endpoints
- Logging settings

**NEVER** hardcode:
- Prompt text in source files
- Model names in code
- API keys or secrets

## Configuration Templates

> **Note**: These are STRUCTURAL TEMPLATES. Adapt content to user's actual project requirements.

### model_config.yaml:
````yaml
models:
  default: claude
  
  claude:
    model_name: claude-sonnet-4-5
    max_tokens: 4096
    temperature: 0.7
    
  gpt:
    model_name: gpt-5
    max_tokens: 4096
    temperature: 0.7

rate_limits:
  requests_per_minute: 50
  tokens_per_minute: 100000

cache:
  enabled: true
  ttl_seconds: 3600
````

### prompts.yaml:
````yaml
system_prompts:
  default: |
    You are a helpful assistant.
    
  analyst: |
    You are a data analyst. 
    Provide structured, factual responses.

templates:
  summarize: |
    Summarize the following text:
    {text}
    
  extract: |
    Extract {entity_type} from:
    {text}
````

### logging_config.yaml:
````yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  detailed:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: data/outputs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  src.llm:
    level: DEBUG
    handlers: [console, file]
    propagate: false
  src.utils:
    level: INFO
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
````

### .env.example:
````bash
# LLM API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Optional: Override config values
LLM_MODEL=claude-sonnet-4-5
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.7

# Logging
LOG_LEVEL=INFO

# Cache
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
````

### .gitignore:
````gitignore
# Environment
.env
.env.local

# Cache
data/cache/
__pycache__/
*.pyc

# Outputs
data/outputs/*.log
data/outputs/*.json

# IDE
.vscode/
.idea/

# Testing
.coverage
htmlcov/
.pytest_cache/

# Dependencies
requirements_check.txt
````

## Required Components

> **Note**: These are IMPLEMENTATION PATTERNS. Use as reference for structure and style, adapt logic to user's requirements.

### Base LLM Client (src/llm/base.py)

Abstract base class for LLM clients with methods:
- `__init__(config)` - Initialize with configuration
- `complete(prompt, **kwargs)` - Generate completion
- `chat(messages, **kwargs)` - Chat conversation
- `count_tokens(text)` - Count tokens

### Rate Limiter (src/utils/rate_limiter.py)

Rate limiter using sliding window algorithm with methods:
- `__init__(requests_per_minute)` - Initialize limiter
- `acquire()` - Acquire permission to make request
- `get_current_usage()` - Get usage statistics

### Token Counter (src/utils/token_counter.py)

Utility for counting and managing tokens with methods:
- `__init__(model)` - Initialize with tokenizer model
- `count(text)` - Count tokens in text
- `truncate(text, max_tokens)` - Truncate text to fit limit
- `split_by_tokens(text, chunk_size)` - Split text into chunks

### Cache (src/utils/cache.py)

File-based cache for LLM responses with methods:
- `__init__(cache_dir, ttl_seconds)` - Initialize cache
- `get(prompt, model)` - Retrieve cached response
- `set(prompt, model, response)` - Store response
- `clear()` - Clear all entries

### Error Handler (src/handlers/error_handler.py)

Error handling with retry logic:
- `APIError` - Base exception for API errors
- `RateLimitError` - Rate limit exception
- `retry_on_error(max_retries, delay, backoff, exceptions)` - Retry decorator

### Logger Setup (src/utils/logger.py)

Logging configuration with methods:
- `setup_logging(config_path)` - Configure from YAML
- `get_logger(name)` - Get logger instance

## Required Tests

> **Note**: These are TEST PATTERNS. Write tests for user's actual implementation, not these examples.

Every GenAI project must include tests for core utilities. Create tests in `tests/` directory.

### Structure:
````
tests/
├── __init__.py
├── conftest.py
├── test_rate_limiter.py
├── test_token_counter.py
├── test_cache.py
└── test_error_handler.py
````

### Test Coverage

Tests should cover:
- **Rate Limiter**: requests under limit, blocking when exceeded, window reset, usage stats
- **Token Counter**: empty strings, simple text, truncation, splitting
- **Cache**: set/get, misses, expiration, different models, clearing, overwriting
- **Error Handler**: first attempt success, retry after failures, exhausted retries, specific exceptions

### requirements.txt:
````
# Core
anthropic>=0.25.0
openai>=1.0.0
tiktoken>=0.6.0
pyyaml>=6.0.0
python-dotenv>=1.0.0

# Code Quality
black>=24.0.0
ruff>=0.4.0

# Dependency Management
pipreqs>=0.5.0
pip-audit>=2.7.0

# Testing
pytest>=8.0.0
pytest-cov>=4.0.0
````

### Run tests command:
````bash
pytest tests/ -v --cov=src --cov-report=term-missing
````

## Mandatory Verification Before Completion

**CRITICAL: NEVER say "done" or "ready" without running these checks.**

### Verification Steps (run in order):
````bash
# 1. Syntax check
python -m py_compile src/**/*.py

# 2. Linting
ruff check .

# 3. Formatting check
black --check .

# 4. Dependency check
pipreqs . --print | grep -v "^#" > /tmp/imports.txt
# Verify all listed packages are in requirements.txt

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run tests
pytest tests/ -v

# 7. Try to import main modules
python -c "from src.llm import base; from src.utils import cache, rate_limiter"
````

### Verification Checklist (must pass ALL):

| Check | Command | Expected |
|-------|---------|----------|
| Syntax | `python -m py_compile file.py` | No output (success) |
| Lint | `ruff check .` | No errors |
| Format | `black --check .` | No reformatting needed |
| Dependencies | `pip install -r requirements.txt` | All packages install |
| Tests | `pytest tests/ -v` | All tests pass |
| Import | `python -c "import src"` | No ImportError |

### On Failure:

1. **Syntax error** → Fix the code, don't just report it
2. **Missing dependency** → Add to requirements.txt, install, verify
3. **Test failure** → Fix the code or test, run again
4. **Import error** → Check file structure, `__init__.py` files, requirements

### Response Format After Verification:
````
Verification completed:
- Syntax: passed
- Lint: passed (or: X warnings, not blocking)
- Tests: 15 passed, 0 failed
- Dependencies: all installed

[Then provide deliverable]
````

### NEVER:
- Say "should work" without running
- Say "you can test by..." — YOU test it
- Skip verification because "it's a small change"
- Assume imports are available without checking requirements.txt

## Validation Checklist

Before completing any GenAI project task, verify:

### Code Style
- [ ] `pyproject.toml` contains Black and Ruff config
- [ ] Code passes `black --check .`
- [ ] Code passes `ruff check .`
- [ ] All functions have type hints
- [ ] All public functions/classes have Google-style docstrings

### Structure & Config
- [ ] Prompts are externalized to YAML or separate files
- [ ] Model parameters are in `config/`
- [ ] `logging_config.yaml` is configured
- [ ] `.env.example` exists with all required variables
- [ ] `.gitignore` excludes `.env`, cache, and outputs
- [ ] Base class exists for LLM clients

### Core Components
- [ ] Rate limiting is implemented
- [ ] Token counting is in place
- [ ] Caching is configured
- [ ] Error handling includes retry logic

### Security & Logging
- [ ] Secrets use environment variables (loaded from `.env`)
- [ ] Request/response logging is enabled
- [ ] No API keys in code or config files

### Dependencies
- [ ] All imports have corresponding entries in `requirements.txt`
- [ ] `pipreqs . --print` matches requirements.txt
- [ ] `pip install -r requirements.txt` succeeds
- [ ] No version conflicts (`pip check`)

### Testing
- [ ] Tests exist for all utilities (rate limiter, cache, token counter, error handler)
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Coverage is acceptable: `pytest --cov=src`

### Verification (run before saying "done")
- [ ] `python -m py_compile` passes on all files
- [ ] `ruff check .` shows no errors
- [ ] `black --check .` shows no changes needed
- [ ] Main modules import without error
- [ ] All tests pass

## README Standards

### Principles:
- **No emojis** — ever
- **No fluff** — every sentence must carry information
- **No repetition** — say it once, say it right
- **Code over prose** — show, don't explain

### Template (only include sections that apply):
````markdown
# Project Name

One sentence: what it does.

## Install
```bash
pip install -r requirements.txt
cp .env.example .env
# add your API keys to .env
```

## Usage
```
[Minimal working example]
```

## Config

Edit `config/model_config.yaml`:
- `model_name`: claude-sonnet-4-5 | gpt-5
- `temperature`: 0.0-1.0
- `max_tokens`: up to 4096

## Project Structure
```
src/
├── llm/        # LLM clients
├── prompts/    # Prompt management  
├── utils/      # Rate limiter, cache, tokens
└── handlers/   # Error handling
```

## Tests
```bash
pytest tests/ -v
```

## License

MIT
````

### Section Rules:

| Section | Rule |
|---------|------|
| Title | Name only, no tagline |
| Description | 1 sentence max, no "This project..." |
| Install | Commands only, comments for non-obvious steps |
| Usage | Minimal working example, no "First you need to..." |
| Config | Table or list of actual options, no defaults explanation |
| Structure | Tree with 1-3 word descriptions |
| Tests | Command only |
| License | One word/line |

### NEVER include:
- Badges (build status, coverage, etc.) unless CI is configured
- "Table of Contents" for READMEs under 100 lines
- "Contributing" section for personal projects
- "Acknowledgments" or "Credits"
- Explanations of what Python/pip/git is
- "Prerequisites: Python 3.11" — put in Install if needed
- Screenshots unless UI exists

### NEVER write:
- "This project is a..."
- "In order to..."
- "You will need to..."
- "Make sure you have..."
- "Don't forget to..."
- "Please note that..."
- "It's important to..."

### Word budget:
- Description: 15 words max
- Each section intro: 0 words (just show the code/config)
- Comments in code: 5 words max

## Antipatterns to Flag

When reviewing code, flag these patterns:

**WRONG:** Hardcoded prompts in code
**CORRECT:** Use template system from YAML

---

**WRONG:** Hardcoded model names
**CORRECT:** Load from config

---

**WRONG:** No retry logic
**CORRECT:** Use retry decorator

---

**WRONG:** No rate limiting
**CORRECT:** Use RateLimiter

---

**WRONG:** Missing type hints and docstrings
**CORRECT:** Add type annotations and Google-style docstrings

---

**WRONG:** Hardcoded API keys
**CORRECT:** Use environment variables

---

**WRONG:** Import without requirements.txt entry
**CORRECT:** Add package to requirements.txt

## Working With User Code

### When User Provides Existing Code:
1. **Analyze first** — understand what the code does
2. **Identify gaps** — what doesn't match standards
3. **Propose changes** — explain what will be improved
4. **Refactor incrementally** — preserve functionality, improve structure
5. **Run verification** — confirm it works

### When User Requests New Feature:
1. **Clarify requirements** — what exactly should it do
2. **Plan implementation** — where it fits in structure
3. **Implement to spec** — user's requirements, your standards
4. **Update requirements.txt** — add any new dependencies
5. **Run verification** — confirm it works
6. **Verify checklist** — ensure compliance

### NEVER:
- Assume templates ARE the task
- Delete working code without explicit permission
- Ignore user's existing architecture choices
- Replace custom logic with generic examples
- Say "done" without running verification
- Leave imports without requirements.txt entries
