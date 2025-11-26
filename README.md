# Generative AI Project Structure

A production-ready project template and coding standards for building robust GenAI applications.


## The Problem

Most Generative AI projects don't fail because of weak models — they fail because the engineering foundation is weak.

Teams often begin with prompts, APIs, and notebooks. But when it's time to scale, integrate, audit, or productionize, everything starts to break.

**What really decides if your AI project moves from demo to product?**

How well it is structured from day one.

---

## What This Repository Provides

**For Humans**: A reference architecture and best practices guide for structuring GenAI projects.

**For AI Agents**: A ready-to-use system prompt that enforces these standards automatically when building or refactoring code.

---

## Quick Start

### For Developers

Browse the `/template` directory for a complete project scaffold you can copy into your own projects.

### For AI Agents

Copy the contents of `AGENT_SYSTEM_PROMPT.md` into your agent's system prompt or custom instructions. The agent will automatically apply these standards when working on GenAI projects.

---

## Project Structure

```
your_project/
├── config/
│   ├── __init__.py
│   ├── model_config.yaml      # Model parameters
│   ├── prompts.yaml           # Prompt templates
│   └── logging_config.yaml    # Logging settings
├── src/
│   ├── llm/                   # LLM client implementations
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract base class
│   │   ├── claude_client.py
│   │   ├── gpt_client.py
│   │   └── utils.py
│   ├── prompts/               # Prompt engineering utilities
│   │   ├── __init__.py
│   │   ├── templates.py
│   │   ├── few_shot.py
│   │   └── chainer.py
│   ├── utils/                 # Core utilities
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
│   ├── test_rate_limiter.py
│   ├── test_token_counter.py
│   ├── test_cache.py
│   └── test_error_handler.py
├── requirements.txt
├── README.md
└── Dockerfile
```

---

## Core Principles

### 1. Configuration Over Hardcoding

All model parameters, prompts, and settings belong in YAML configuration files — not scattered throughout your codebase.

```yaml
# config/model_config.yaml
models:
  default: claude
  claude:
    model_name: claude-sonnet-4-20250514
    max_tokens: 4096
    temperature: 0.7
```

### 2. Modular LLM Clients

Separate client implementation for each LLM provider, built on a common interface. Switch providers without rewriting your application.

### 3. Production Utilities

Every GenAI project needs these from day one:

| Utility | Purpose |
|---------|---------|
| Rate Limiter | Prevent API throttling |
| Token Counter | Track usage and costs |
| Response Cache | Reduce redundant API calls |
| Error Handler | Retry logic with exponential backoff |

### 4. Separation of Concerns

- `config/` — All configuration, no code
- `src/llm/` — LLM interactions only
- `src/utils/` — Reusable utilities
- `src/handlers/` — Error handling and middleware
- `examples/` — Usage demonstrations
- `tests/` — Test coverage

### 5. Testing

Production code requires tests. All core utilities must have test coverage:

| Component | What to test |
|-----------|--------------|
| Rate Limiter | Request counting, blocking when limit exceeded, window reset |
| Token Counter | Counting accuracy, truncation, chunking |
| Cache | Set/get, expiration, cache miss, clearing |
| Error Handler | Retry attempts, exception filtering, backoff timing |

Run tests:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Common Antipatterns

| Antipattern | Problem | Solution |
|-------------|---------|----------|
| Hardcoded prompts | Can't iterate without code changes | Store in YAML files |
| Single monolithic file | Unmaintainable, untestable | Modular architecture |
| No rate limiting | API throttling, failed requests | Implement rate limiter |
| No token tracking | Unpredictable costs | Count tokens per request |
| Secrets in code | Security vulnerability | Use environment variables |
| No error handling | Silent failures | Retry with backoff |
| No tests | Regressions, broken deploys | Test all utilities |

---

## Production Considerations

This template provides the foundation. For production deployment, also consider:

**Security and Governance**
- Secret management (Vault, AWS Secrets Manager)
- PII detection and handling
- Content moderation and guardrails

**Observability**
- Token usage analytics
- Latency monitoring
- Cost tracking dashboards
- Error rate alerting

---

## Repository Contents

```
.
├── README.md                    # This file
├── AGENT_SYSTEM_PROMPT.md       # Copy-paste prompt for AI agents
├── template/                    # Complete project scaffold
│   ├── config/
│   ├── src/
│   ├── examples/
│   └── ...
└── docs/
    └── detailed_guide.md        # Extended documentation
```

---

## Usage with AI Agents

This repository includes a system prompt designed for AI coding agents (Claude, GPT, Cursor, etc.). 

**To use:**

1. Open `AGENT_SYSTEM_PROMPT.md`
2. Copy the entire contents
3. Add to your agent's system prompt or custom instructions

The agent will then automatically:
- Apply this project structure to new GenAI projects
- Refactor existing code to follow these patterns
- Flag antipatterns and suggest fixes

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
