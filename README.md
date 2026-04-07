---
title: OpenEnv DataClean Agent
emoji: "🧹"
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# DataCleanEnv — Meta OpenEnv Hackathon 2026

## What This Agent Does

**DataCleanEnv trains AI agents to automatically clean messy real-world data.**

Give it a dirty CSV file. The agent reads it, finds problems (duplicates, missing values, bad dates, wrong formats), writes Python code to fix them, and submits a clean file. It gets scored 0.0–1.0 on how well it did.

### Real Example

**Input (messy):**
```
order_id,customer_name,order_date,amount,email
1001,  JOHN DOE  ,2024-01-15,  100.50  ,john@example.com
1002,jane smith,01/16/2024,$150.75,JANE@EXAMPLE.COM
1003,Bob Johnson  ,2024-01-17,200,bob@example
```

**What the agent does:**
1. Reads the file → sees the mess
2. Runs Python → identifies issues (inconsistent dates, casing, invalid emails, extra whitespace, $ signs in amounts)
3. Writes cleaning code → pandas pipeline to fix everything
4. Submits cleaned file → gets graded

**Output (clean):**
```
order_id,customer_name,order_date,amount,email
1001,John Doe,2024-01-15,100.50,john@example.com
1002,Jane Smith,2024-01-16,150.75,jane@example.com
1003,Bob Johnson,2024-01-17,200.00,bob@example.com
```

### Three Difficulty Levels

| Level | What It Tests |
|-------|--------------|
| **Easy** (5 rows) | Remove duplicates, handle missing values |
| **Medium** (20 rows) | Fix dates, casing, emails, whitespace, data types |
| **Hard** (50 rows) | Missing IDs, negative values, mixed formats, units, normalization |

---

## How We Win This Hackathon

### The Judging Criteria (and how we max each one)

#### 1. Real-World Utility — 30% → Target: 28/30

**Why we score high:**
- Data cleaning is the #1 time sink for data scientists (80% of their time)
- Every company has this problem — finance, healthcare, e-commerce, all of them
- Directly applicable to real tools: dbt, Great Expectations, pandas pipelines
- Judges will immediately understand the value

**What to show judges:**
- "This is what real messy data looks like" → show the CSV
- "This is what the agent produces" → show the clean output
- "This saves hours of manual work" → connect to real business impact

#### 2. Task & Grader Quality — 25% → Target: 24/25

**Why we score high:**
- **100% deterministic grading** — no fuzzy text matching, no subjective evaluation
- Ground truth CSV exists for every task → exact comparison
- Multi-metric scoring: rows (20%) + columns (20%) + cells (60%)
- Three clear difficulty levels showing progression
- Can't game the system — you either clean the data or you don't

**Competitive edge:** Most submissions use text-based grading (is the answer "close enough"?). Ours is exact. Judges love this because it's fair and reproducible.

#### 3. Environment Design — 20% → Target: 18/20

**Why we score high:**
- Clean 4-tool interface: read, execute python, write, submit
- Isolated workspace per episode (no cross-contamination)
- Python sandbox with pandas/numpy (real data science tools)
- Follows OpenEnv MCP pattern exactly (same architecture as FinQA)
- Max step limits prevent infinite loops

#### 4. Code Quality & Spec Compliance — 15% → Target: 14/15

**Why we score high:**
- Passes `openenv validate`
- Dockerfile works out of the box
- `inference.py` in root directory (hackathon requirement)
- Clean separation: models, tools, environment, rewards, app
- Follows FinQA reference implementation pattern

#### 5. Creativity & Novelty — 10% → Target: 8/10

**Why we score high:**
- First data cleaning RL environment in the OpenEnv ecosystem
- Novel application of tool-calling agents to data engineering
- Real CSV files with realistic planted issues (not synthetic toy data)

### Total Estimated Score: 92/100 → Top 5%

---

### What Other Teams Will Build (and why we beat them)

| What they'll build | Why we beat them |
|-------------------|-----------------|
| Game environments (Snake, Connect4) | Ours has real-world utility, theirs is a toy |
| Simple Q&A environments | Our grading is deterministic, theirs is fuzzy |
| Code generation envs | Ours is more focused and easier to demonstrate |
| SQL query generators | Already exists (FinQA), we're novel |

### Demo Strategy for Judges

1. **Show the messy file** (10 seconds) — "Look at this real-world messy data"
2. **Run the agent** (30 seconds) — "Watch it identify and fix issues"
3. **Show the clean output** (10 seconds) — "Perfect match with ground truth"
4. **Show the score** (5 seconds) — "Reward: 0.95 — nearly perfect"
5. **Show all 3 levels** (30 seconds) — "Easy → Medium → Hard progression"
6. **Show the RL training** (optional) — "And here's how we train it to get better"

---

## Quick Start

```bash
# Set environment variables
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=your_model

# Build and run
docker build -f data_clean_env/server/Dockerfile -t data-clean-env:latest .
docker run -p 8000:8000 data-clean-env:latest

# In another terminal
python inference.py
```

## RL Training (Free GPU)

Open `data_clean_env/RL_Training_Colab.ipynb` in Google Colab → Runtime → GPU → Run all.

Works on:
- **Google Colab** (free T4 GPU, 16GB VRAM)
- **Kaggle** (free P100, 30hrs/week)
- **Your friend's RTX 4050** (6GB VRAM, QLoRA on Qwen2.5-0.5B)

---

## Project Structure

```
data_clean_env/
├── tasks/              # Messy + clean CSV pairs for 3 difficulty levels
├── server/
│   ├── environment.py  # Main env with reset()/step()
│   ├── tools.py        # 4 MCP tools (read/python/write/submit)
│   ├── rewards.py      # Deterministic CSV grading
│   ├── app.py          # FastAPI server
│   └── Dockerfile      # HF Spaces deployment
├── client.py           # MCP client
├── models.py           # State definition
├── openenv.yaml        # OpenEnv spec
├── pyproject.toml      # Dependencies
└── RL_Training_Colab.ipynb  # Free GPU training notebook

inference.py            # Root-level inference (hackathon requirement)
```

---

**Built for:** Meta PyTorch & Hugging Face OpenEnv Hackathon 2026
**Stack:** OpenEnv, FastMCP, FastAPI, pandas, Docker
