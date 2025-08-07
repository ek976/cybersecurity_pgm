# Cybersecurity PGM Project Structure

## Core Models Architecture

### 1. **timing_model.py** - Binary Timing Attack
**Purpose**: Demonstrate timing side-channel attacks where response time reveals correctness

**Key Components**:
- Binary nodes: G_i ∈ {0,1} (incorrect/correct)
- Timing observations: T_i ∈ {0,1,2} (short/medium/long)
- No character prediction, only position correctness

**Core Functions**:
```python
def measure_time(user_input, secret)  # Simulates timing delay
def collect_timing_data(secret)       # Gathers timing measurements
def build_timing_model()              # Creates binary PGM
def run_timing_inference(model, timing_classes)  # Infers correct positions
def suggest_binary_guess(posteriors)  # Returns "1011101..." string
```

### 2. **wordlike_model.py** - Character Prediction
**Purpose**: Predict password characters using n-gram language models

**Key Components**:
- Character nodes: G_i ∈ CHARSET (actual characters)
- Uses bigram/trigram probabilities from training data
- No timing information

**Core Functions**:
```python
def build_wordlike_model(words, use_ngrams)  # Creates character PGM
def run_wordlike_inference(model, observed)   # Predicts missing chars
def suggest_wordlike_guess(posteriors, observed)  # Returns "vegetable"
```

### 3. **hybrid_model.py** - Combined Attack
**Purpose**: Combine timing and character prediction for sophisticated attack

**Key Components**:
- Character nodes: G_i ∈ CHARSET
- Timing nodes: T_i ∈ {0,1,2}
- Position correctness nodes: C_i ∈ {0,1}
- Edge structure: G_i → C_i → T_i (character affects correctness affects timing)

**Core Functions**:
```python
def build_hybrid_model(words, use_ngrams)     # Combined PGM
def run_hybrid_inference(model, observed_chars, timing_classes)
def suggest_hybrid_guess(posteriors)          # Best of both worlds
```

## Visualization Files

### 1. **timing_plot.py**
```python
def plot_timing_pgm_structure()    # Show G→T causal structure
def plot_timing_posteriors()       # Display P(correct) for each position
def plot_timing_operational()      # Show timing bins vs positions
```

### 2. **wordlike_plot.py**
```python
def plot_wordlike_pgm_structure()  # Show G₁→G₂→G₃ chain
def plot_wordlike_posteriors()     # Display character probabilities
def plot_wordlike_operational()    # Show top-k character predictions
```

### 3. **hybrid_plot.py**
```python
def plot_hybrid_pgm_structure()    # Show full G→C→T network
def plot_hybrid_posteriors()       # Combined probability display
def plot_hybrid_operational()      # Timing + character predictions
```

## File Dependencies

```
common_words.py
    ↓
wordlike_model.py
    ↓
hybrid_model.py ← timing_model.py
    ↓
Plot files (independent visualizations)
```
