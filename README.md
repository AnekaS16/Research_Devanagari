# Hindi Lexical Sensitivity Research System

## Research objective

Study how contextual lexical contrasts, synonym substitutions, and realistic spelling errors in Hindi Devanagari affect subword tokenization and pretrained masked-language-model behavior.

## Research principles

- Preserve the original dataset.
- Never silently modify human annotations.
- Keep processed data and experiment outputs separate.
- Record experiment configurations and software versions.
- Do not interpret language-model scores as direct measures of grammatical correctness.
- Do not report results until experiments have actually run.

## Setup & Installation

### 1. Python Environment Setup
Recommended: Python 3.10 to 3.12.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch**: If your teammate has an NVIDIA GPU and wants CUDA acceleration, install CUDA-enabled PyTorch first from [pytorch.org](https://pytorch.org) (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`), then run `pip install -r requirements.txt`.

### 2. Verify Installation
Run the test suite to ensure all components and models work as expected:
```bash
pytest
```

### 3. Interactive Dashboard (Optional)
The project includes a React/Vite visualization dashboard:
```bash
cd dashboard
npm install
npm run dev
```

