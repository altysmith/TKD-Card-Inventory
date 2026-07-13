# TKD Card Inventory

Desktop Pokémon card inventory application built with Python, PySide6, SQLite, and the Pokémon TCG API.

## Current MVP

- Desktop application with PySide6
- Search Pokémon cards by name and optional collector number
- Preview matching cards
- Add selected cards to a local SQLite inventory
- Automatically increment duplicate quantities
- View current inventory
- Export inventory to CSV or Excel

## Setup

1. Install Python 3.11 or newer.
2. Clone this repository.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Copy `.env.example` to `.env` and add your Pokémon TCG API key:

```env
POKEMON_TCG_API_KEY=your_key_here
```

6. Run the application:

```powershell
python -m src.main
```

## Local data

The application stores its SQLite inventory database in the user's local application-data directory, not inside the Git repository. Exported CSV and Excel files are written to a folder selected by the user.

## Planned next steps

- Image upload and card cropping
- OCR-assisted name and collector-number recognition
- Live webcam capture
- Scan-session history and undo
- Improved matching and confidence scoring
