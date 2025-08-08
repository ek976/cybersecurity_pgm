import requests
from pathlib import Path

# Define allowed charset if not already defined elsewhere
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"

def download_rockyou_subset(top_n=10000, cache_dir="./wordlists"):
    """
    Download top N passwords from RockYou breach dataset.
    Uses SecLists repository (curated security wordlists).
    """
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = Path(cache_dir) / f"rockyou_top{top_n}.txt"
    
    if cache_file.exists():
        print(f"Loading cached {cache_file}")
        with open(cache_file, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.strip() for line in f if line.strip()]
    
    # SecLists provides cleaned common passwords
    urls = {
        100: "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/10-million-password-list-top-100.txt",
        1000: "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/10-million-password-list-top-1000.txt",
        10000: "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/10-million-password-list-top-10000.txt",
    }
    
    # Find appropriate URL
    url_key = min(k for k in urls.keys() if k >= top_n)
    url = urls.get(url_key)
    
    if url:
        print(f"Downloading top {url_key} passwords from SecLists...")
        try:
            response = requests.get(url)
            passwords = response.text.strip().split('\n')[:top_n]
            
            # Cache for future use
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(passwords))
            
            print(f"Cached {len(passwords)} passwords to {cache_file}")
            return passwords
        except Exception as e:
            print(f"Error downloading passwords: {e}")
            return []
    
    return []

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
    
    try:
        response = requests.get(url)
        all_words = response.text.strip().split('\n')
        
        words = [w.lower() for w in all_words if min_length <= len(w) <= max_length]
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(words))
        
        print(f"Cached {len(words)} words to {cache_file}")
        return words
    except Exception as e:
        print(f"Error downloading words: {e}")
        return []

def create_hybrid_wordlist(n_passwords=2500, n_words=2500, add_variations=True):
    """
    Create a hybrid list combining common passwords and dictionary words.
    """
    passwords = download_rockyou_subset(n_passwords)
    words = download_english_words()[:n_words]
    
    combined = list(set(passwords + words))
    
    if add_variations:
        variations = []
        common_suffixes = ['1', '123', '!', '2024', '2025', '@']
        
        for word in combined[:500]:  # Add variations for top 500
            if word and word[0].isalpha():
                variations.append(word.capitalize())
            
            for suffix in common_suffixes:
                if len(word) + len(suffix) <= 12:
                    variations.append(word + suffix)
        
        combined.extend(variations)
    
    # Filter to match charset and length constraints
    filtered = []
    for word in combined:
        if word and 3 <= len(word) <= 10 and all(c in CHARSET for c in word):
            filtered.append(word)
    
    print(f"Final hybrid wordlist: {len(filtered)} entries")
    return filtered

def get_training_words(source="hybrid", limit=5000):
    """
    Get training words from various sources.
    This is the main function that the models import.
    
    Parameters:
    -----------
    source : str
        'passwords' - Common passwords only
        'english' - English dictionary only  
        'hybrid' - Mix of both (recommended)
        'default' - Small default set for testing
    limit : int
        Maximum number of words to return
    """
    if source == "passwords":
        words = download_rockyou_subset(limit)
    elif source == "english":
        words = download_english_words()[:limit]
    elif source == "hybrid":
        words = create_hybrid_wordlist(n_passwords=limit//2, n_words=limit//2)
    elif source == "default":
        # Small default set for testing
        words = [
            "password", "hunter2", "Welcome123", "Admin@2021", 
            "vegetable", "Monkey", "Dragon42", "letmein",
            "football", "baseball", "master", "michael",
            "shadow", "ashley", "football", "jesus",
            "ninja", "mustang", "password1", "password123"
        ]
    else:
        print(f"Unknown source: {source}, using default")
        words = ["password", "hunter2", "Welcome123", "vegetable"]
    
    # Filter for valid charset and length
    valid_words = []
    for word in words:
        if word and 3 <= len(word) <= 10:
            # Check if word contains only valid characters
            if all(c in CHARSET for c in word):
                valid_words.append(word)
    
    # Ensure we have at least some words
    if not valid_words:
        print("Warning: No valid words found, using defaults")
        valid_words = ["password", "hunter", "admin", "test"]
    
    return valid_words[:limit]

# For backward compatibility
def create_english_wordlist(n_words=5000, add_variations=True):
    """
    Create a filtered list of English words with optional variations.
    Kept for backward compatibility.
    """
    words = download_english_words()[:n_words]
    
    if add_variations:
        variations = []
        common_suffixes = ['1', '123', '!', '2024', '2025', '@']
        
        for word in words[:1000]:  # only top 1000 get variations
            if word and word[0].isalpha():
                variations.append(word.capitalize())
            for suffix in common_suffixes:
                if len(word) + len(suffix) <= 12:
                    variations.append(word + suffix)
        
        words.extend(variations)
    
    filtered = [w for w in words if w and len(w) <= 10 and all(c in CHARSET for c in w)]
    
    print(f"Final wordlist: {len(filtered)} entries")
    return filtered