import os
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "Bridgerton S1 Episode 1.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GLOSSARY_PATH = os.path.join(BASE_DIR, "data", "bridgerton_glossary.json")

# === Audio Extraction ===
AUDIO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "audio.wav")
AUDIO_SAMPLE_RATE = 16000  # Whisper expects 16kHz

# === Transcription ===
WHISPER_MODEL = "medium"  # 769M params, best CPU accuracy/speed
WHISPER_LANGUAGE = "en"   # Source audio is English

# === Claude API ===
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
TRANSLATION_BATCH_SIZE = 10  # Segments per API call

# === Subtitle Timing ===
MAX_CHARS_PER_LINE = 42        # Industry standard line length
MAX_LINES_PER_SUBTITLE = 2    # Max 2 lines per subtitle
MIN_DISPLAY_DURATION_MS = 1000 # Minimum 1 second on screen
MAX_DISPLAY_DURATION_MS = 7000 # Maximum 7 seconds on screen
READING_SPEED_CPS = 14         # Hebrew characters per second (standard)
GAP_BETWEEN_SUBS_MS = 80       # Minimum gap between subtitles

# === RTL Unicode Characters ===
RLM = "\u200F"  # Right-to-Left Mark
RLE = "\u202B"  # Right-to-Left Embedding
PDF = "\u202C"  # Pop Directional Formatting

# === Output Paths ===
SRT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Bridgerton_S1E1_Hebrew.srt")
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Bridgerton_S1E1_data.csv")
TOOLS_DOC_PATH = os.path.join(OUTPUT_DIR, "models_and_tools.md")
