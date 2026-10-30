from pathlib import Path

from dotenv import load_dotenv

# settings such as GEMINI_API_KEY can live in a .env file at the repo root (it is gitignored);
# variables already set in the environment take precedence
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
