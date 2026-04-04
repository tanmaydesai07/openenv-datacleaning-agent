
Join Discord

Help

Log out

Registration

14th March - 3rd April

Declaration

Before R1

Prepare

Now - 25th March

Round 1

25th March - 8th April

Results

10th April

Finale

25th-26th April

Welcome Tanmay Desai!

tanmay2004desai@gmail.com
Copy

Join the Discord Community

All announcements, mentor access, and team matching happens here.


Join Discord
QUICK TOGGLe

Team form Submission

Preparatory Course

Start Assessment

FAQs

step 1

How will you compete?

Choose solo or team before you can start the assessment

Step 1 Complete
Team: Code Blizzards

👤
Rhugved Dangui
rhugveddangui07@gmail.com
Accepted
👤
Tanmay Desai
tanmay2004desai@gmail.com
Team Lead
👤
Fayaz Khan
fayaz.khanxid411@gmail.com
Accepted
🔒
Team is permanently locked. Changes are not allowed after confirmation.

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp: Build Your First RL Environment

Live walkthrough to submit a strong Round 1 entry

timing

8:00 PM Onwards

Wednesday, 1st April

Host


Ben Burtenshaw

Community Education in AI at Hugging Face


Pulkit Aneja

Scaler Instructor

Watch Recording

PROBLEM STATEMENT

Round 1 — Problem Statement

The Task

Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard  step() / reset() / state()  API.

Key Requirements at a Glance

Must simulate a real-world task (not games or toys)

Implement full OpenEnv spec: typed models, step()/reset()/state(), openenv.yaml

Minimum 3 tasks with agent graders (easy → medium → hard, scores/reward 0.0–1.0)

Meaningful reward function with partial progress signals

Baseline inference script with reproducible scores

Deploy to Hugging Face Spaces + working Dockerfile

README with environment description, action/observation spaces, setup instructions

Functional Requirements

Real-world task simulation

The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.

OpenEnv spec compliance

Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) → returns observation, reward, done, info. reset() → returns initial observation. state() → returns current state. openenv.yaml with metadata. Tested via openenv validate.

Minimum 3 tasks with agent graders

Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.

Meaningful reward function

Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).

Baseline inference script

Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (OPENAI_API_KEY). Produces a reproducible baseline score on all 3 tasks.

Detailed Requirements

Non-Functional Requirements

Deploys to a Hugging Face Space

Environment must run as a containerized HF Space tagged with openenv.

Containerized execution

Must include a working Dockerfile. The environment should start cleanly with docker build + docker run.

Documentation

README must include: environment description and motivation, action and observation space definitions, task descriptions with expected difficulty, setup and usage instructions, baseline scores.

Parameter

Weight

Description

Real-world utility

30%

Does the environment model a genuine task? Would someone actually use this to train or evaluate agents?

Task & grader quality

25%

Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression?

Environment design

20%

Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries.

Code quality & spec compliance

15%

Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works.

Creativity & novelty

10%

Novel problem domain, interesting mechanics, clever reward design, original approach.

Scoring Breakdown

Real-world utility (30%)

•  0–5: Toy/artificial problem with no practical application

•  6–15: Valid domain but shallow modeling of the real task

•  16–25: Good domain modeling, would be useful for agent evaluation

•  26–30: Excellent — fills a real gap, immediate value for the RL/agent community

Task & grader quality (25%)

•  3+ tasks with difficulty range?

•  Graders produce scores between 0.0–1.0?

•  Graders deterministic and reproducible?

•  Hard task genuinely challenges frontier models?

Environment design (20%)

•  reset() produces clean state?

•  Action/observation types well-designed and documented?

•  Reward function provides useful varying signal (not just sparse)?

•  Episode boundaries sensible?

Code quality & spec compliance (15%)

•  openenv validate passes?

•  docker build && docker run works?

•  HF Space deploys and responds?

•  Baseline script runs and reproduces scores?

Creativity & novelty (10%)

•  Domain we haven’t seen in OpenEnv before?

•  Reward design has interesting properties?

•  Clever mechanics that make the environment engaging?

Evaluation Criteria

Phase 1: Automated Validation

Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.

Phase 2: Agentic Evaluation

Scored — baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.

Phase 3: Human Review

Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

Disqualification Criteria

Environment does not deploy or respond

Plagiarized or trivially modified existing environments

Graders that always return the same score

No baseline inference script

How Judging works

Pre-Submission Checklist  — all must pass or you're disqualified

HF Space deploys

Automated ping to the Space URL — must return 200 and respond to reset()

OpenEnv spec compliance

Validate openenv.yaml, typed models, step()/reset()/state() endpoints

Dockerfile builds

Automated docker build on the submitted repo

Baseline reproduces

Run the submitted inference script — must complete without error and produce scores

3+ tasks with graders

Enumerate tasks, run each grader, verify scores/reward in 0.0–1.0 range

Mandatory Additional Instructions

Before submitting, ensure the following variables are defined in your environment configuration:

API_BASE_URL   The API endpoint for the LLM.

MODEL_NAME     The model identifier to use for inference.

HF_TOKEN       Your Hugging Face / API key.

The inference script must be named `inference.py` and placed in the root directory of the project

Participants must use OpenAI Client for all LLM calls using above variables

Participants must emit structured stdout logs strictly following the [START], [STEP], and [END] format defined in the sample inference.py provided below. Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring. Refer to the Sample Inference Script for the complete format specification and examples.

Infra Restrictions

Runtime of inference script should be less than 20min 

Make sure your env and inference can run on a machine with vcpu=2, memory=8gb

Validator

Run the pre-submission validation script before submitting

Sample Inference Script

Pre Validation Script

Submission window opens on 28th March

Deadline: 8 Apr 11:59 PM


Submit your Assessment
→
Study material

Preparatory Course

4 modules · ~3.5 hours 

Each module: read the README first, then open the notebook in Colab. No local setup needed.

 Module 1: Why OpenEnv?

ESSENTIAL FOR ROUND 1

45 min

Module 2: Using Existing Environments

ESSENTIAL FOR ROUND 1

50 min

 Module 3: Deploying Environments

ESSENTIAL FOR ROUND 1

45 min

Module 4: Building Your Own Environment

 MOST IMPORTANT FOR ROUND 1

60 min

View full course repository

GUIDE

Round 1 Guide

What to Expect

Prerequisites

How to Submit

Install before April 1st.

Python 3.10+

Install 3.10, 3.11, or 3.12.

$
python --version
Copy
Git + GitHub account

Push your submission to GitHub or HF.

$
git --version
Copy
Hugging Face CLI

Deploy to HF Spaces.

$
pip install huggingface_hub --version
Copy
$
huggingface-cli login
Copy
OpenEnv

The framework.

$
pip install openenv-core
Copy
Google Colab

Prep course runs in Colab. Free tier works.

$
pip install openenv-core
Copy
OpenEnv

The framework.

→ colab.research.google.com
Copy
Docker

Isolated container testing.

docker --version
Copy
Recommended

VS Code

Best Python + Docker support

Step 2

Submit your Assessment

Complete Step 1 first

Problem Statement is live. Build and submit.

Round 1 begins 

Submission window opens on 28th March

Deadline: 8 Apr 11:59 PM


Submit your Assessment
→
NOTE: Only team leaders can make the final submission.

FAQs

Frequently Asked Questions













Need help? Reach out to us

help_openenvhackathon@scaler.com

Contact Support

submission Deadline: 8th April 11:59 PM


Submit your Assessment
→
How to Submit?

Hi! What can I help you with?