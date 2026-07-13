from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class PokemonTCGClient:
    BASE_URL = "https://api.pokemontcg.io/v2/cards"

    def __init__(self) -> None:
        self.api_key = os.getenv("POKEMON_TCG_API_KEY", "").strip()

    def search_cards(self, name: str, number: str = "") -> list[dict[str, Any]]:
        query_parts: list[str] = []
        if name.strip():
            safe_name = name.strip().replace('"', "")
            query_parts.append(f'name:"{safe_name}*"')
        if number.strip():
            safe_number = number.strip().split("/")[0]
            query_parts.append(f'number:"{safe_number}"')

        if not query_parts:
            return []

        headers = {"X-Api-Key": self.api_key} if self.api_key else {}
        response = requests.get(
            self.BASE_URL,
            params={
                "q": " ".join(query_parts),
                "pageSize": 50,
                "orderBy": "name,set.releaseDate",
                "select": "id,name,number,rarity,set,images",
            },
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()

        cards: list[dict[str, Any]] = []
        for raw in response.json().get("data", []):
            card_set = raw.get("set", {})
            images = raw.get("images", {})
            printed_total = card_set.get("printedTotal") or card_set.get("total")
            display_number = raw.get("number", "")
            if printed_total:
                display_number = f"{display_number}/{printed_total}"

            cards.append(
                {
                    "id": raw["id"],
                    "name": raw.get("name", "Unknown"),
                    "number": display_number,
                    "raw_number": raw.get("number", ""),
                    "rarity": raw.get("rarity", ""),
                    "set_name": card_set.get("name", "Unknown Set"),
                    "set_code": card_set.get("ptcgoCode") or card_set.get("id", ""),
                    "image_url": images.get("small") or images.get("large", ""),
                }
            )
        return cards
