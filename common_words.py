import requests
from pathlib import Path

# Define allowed charset if not already defined elsewhere
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%"

def download_english_words(min_length=4, max_length=12, cache_dir="./wordlists"):
    """
    Download and cache common English words (filtered by length).
    Source: https://github.com/dwyl/english-words
    """
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = Path(cache_dir) / f"english_words_{min_length}_{max_length}.txt"
    
    if cache_file.exists():
        print(f"Loading cached {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    print("Downloading English word list...")
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    response = requests.get(url)
    all_words = response.text.strip().split('\n')
    
    words = [w.lower() for w in all_words if min_length <= len(w) <= max_length]
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))
    
    print(f"Cached {len(words)} words to {cache_file}")
    return words

def create_english_wordlist(n_words=5000, add_variations=True):
    """
    Create a filtered list of English words with optional variations.
    """
    words = download_english_words()[:n_words]
    
    if add_variations:
        variations = []
        common_suffixes = ['1', '123', '!', '2024', '2025', '@']
        
        for word in words[:1000]:  # only top 1000 get variations
            if word[0].isalpha():
                variations.append(word.capitalize())
            for suffix in common_suffixes:
                if len(word) + len(suffix) <= 12:
                    variations.append(word + suffix)
        
        words.extend(variations)
    
    filtered = [w for w in words if len(w) <= 10 and all(c in CHARSET for c in w)]
    
    print(f"Final wordlist: {len(filtered)} entries")
    return filtered
