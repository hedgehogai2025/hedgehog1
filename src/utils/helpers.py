import os
import json
from datetime import datetime, timedelta

def ensure_directory_exists(directory):
    """Ensure a directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_json(data, filepath):
    """Save data as JSON to the specified filepath."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    """Load JSON data from the specified filepath."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def format_large_number(number):
    """Format large numbers in a readable way (e.g., 1.5M, 2.3B)."""
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.1f}K"
    else:
        return str(number)

def get_time_ago(timestamp):
    """Convert timestamp to 'X time ago' format."""
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            return "unknown time ago"
    
    if not isinstance(timestamp, datetime):
        return "unknown time ago"
    
    now = datetime.now()
    diff = now - timestamp
    
    if diff.days > 365:
        return f"{diff.days // 365}y ago"
    elif diff.days > 30:
        return f"{diff.days // 30}mo ago"
    elif diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600}h ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60}m ago"
    else:
        return f"{diff.seconds}s ago"

def truncate_text(text, max_length=280):
    """Truncate text to fit within Twitter's character limit."""
    if len(text) <= max_length:
        return text
    
    # Try to truncate at a sentence boundary
    sentences = text.split('. ')
    result = ""
    for sentence in sentences:
        if len(result) + len(sentence) + 2 <= max_length - 3:  # -3 for "..."
            result += sentence + ". "
        else:
            break
    
    # If we couldn't even fit one sentence, just truncate
    if not result:
        result = text[:max_length-3]
    
    return result.strip() + "..."