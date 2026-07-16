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
- Live webcam capture with burst-frame quality selection
- Local title, set-code, and collector-number OCR
- Catalog-aware correction of incomplete or damaged OCR text

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

The application stores its SQLite inventory database in the user's local application-data directory, not inside the Git repository. Scanner outcomes are appended to `scan_history.csv` in that same directory for accuracy analysis; card images are not saved. Exported CSV and Excel files are written to a folder selected by the user.

## Scanner design

The scanner works offline after the catalog is downloaded. It straightens a
detected card, selects the clearest title and identifier regions from a short
camera burst, reads them independently, and ranks local
catalog candidates using all available evidence. A result is collapsed to one
card only when multiple compatible clues make it decisive. Partial collector
numbers and printed totals never force automatic selection on their own;
ambiguous prints remain available for manual review.

Catalog resolution is implemented in a standalone, non-Qt `CardMatcher` so
matching strategies and confidence thresholds can be tested independently of
the camera and desktop interface.

Foil glare can still obscure printed details. A fixed camera position with
soft, diffuse lighting gives the most repeatable results.
