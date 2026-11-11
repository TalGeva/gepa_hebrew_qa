"""Project paths and experiment settings."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HEQ_DIR = DATA_DIR / "heq"
RUNS_DIR = PROJECT_ROOT / "runs"

# Official HeQ v1.1 splits (https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset).
# train has one gold span per question; val/test have several valid spans per question.
HEQ_FILES = {
    "train": HEQ_DIR / "train_v1.1.json",
    "val": HEQ_DIR / "val_v1.1.json",
    "test": HEQ_DIR / "test_v1.1.json",
}

# Main experiment: Hebrew abstractive QA (free-text answers about a document).
ABSTRACTIVE_FILE = DATA_DIR / "abstractive" / "abstractive_qa_for_llm_eval.json"


@dataclass(frozen=True)
class PilotConfig:
    """Stage 1 pilot on HeQ: sample sizes are drawn from the matching official split."""

    n_train: int = 100
    n_val: int = 100
    n_test: int = 300
    seed: int = 42
    include_unanswerable: bool = False


@dataclass(frozen=True)
class AbstractiveSplitConfig:
    """Train/val sizes in questions; the rest is test. Splits are drawn by document."""

    n_train: int = 30
    n_val: int = 30
    seed: int = 42


@dataclass(frozen=True)
class LMConfig:
    """A model and its sampling settings. OpenRouter by default; api_base points at any
    OpenAI-compatible server instead (for example vLLM serving Hebatron on the cluster)."""

    model: str
    temperature: float
    max_tokens: int
    api_base: str | None = None
    top_p: float | None = None
    # extra JSON fields for the request body, e.g. {"chat_template_kwargs": {"enable_thinking": False}}
    # to switch a reasoning model's chat template (Hebatron/Nemotron: thinking is on by default)
    extra_body: dict | None = None


# Task model: Qwen 3.6, ~30B class.
TASK_LM = LMConfig(model="qwen/qwen3.6-35b-a3b", temperature=0.0, max_tokens=4000)
# Reflection model: reads traces and feedback and proposes new instructions. The GEPA docs
# recommend a strong model with a high temperature and a large output budget.
REFLECTION_LM = LMConfig(model="anthropic/claude-sonnet-5.5", temperature=1.0, max_tokens=16000)
# Judge candidates for the abstractive answers (stage 2). Both are outside the Qwen family;
# the one that agrees best with human scores is frozen for the rest of the project.
JUDGE_LMS = {
    "sonnet": LMConfig(model="anthropic/claude-sonnet-5.5", temperature=0.0, max_tokens=4000),
    "gemma": LMConfig(model="google/gemma-4-31b-it", temperature=0.0, max_tokens=4000),
}
# Frozen 2026-11-04 after validation against the dataset's human ratings
# (runs/2026-11-04_12-11-27_judge_validation). Gemma almost never scored below 2.
JUDGE_LM = JUDGE_LMS["sonnet"]


@dataclass(frozen=True)
class GEPAConfig:
    auto: str = "light"  # GEPA budget preset: "light", "medium" or "heavy"
    feedback: str = "hebrew"  # "hebrew": diagnostic Hebrew feedback; "score": the score only (RQ2)
    reflection_minibatch_size: int = 3
    seed: int = 0
    num_threads: int = 8
    # extra guidance appended to every reflection prompt (steers how the reflection LM rewrites); None = GEPA default
    reflection_guidance: str | None = None
