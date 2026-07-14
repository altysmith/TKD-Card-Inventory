from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

SET_CODE_ALIASES = {
    "sv3pt5": "MEW",
}


def normalize_card_category(supertype: str, subtypes: list[str]) -> str:
    subtype_lookup = {value.casefold() for value in subtypes}
    if supertype.casefold() == "pokémon":
        return "Pokémon"
    if supertype.casefold() == "energy":
        return "Special Energy" if "special" in subtype_lookup else "Basic Energy"
    if "supporter" in subtype_lookup:
        return "Supporter"
    if "stadium" in subtype_lookup:
        return "Stadium"
    if "pokémon tool" in subtype_lookup or "tool" in subtype_lookup:
        return "Pokémon Tool"
    if "item" in subtype_lookup:
        return "Item"
    return supertype or "Other"


class PokemonTCGClient:
    BASE_URL = "https://api.pokemontcg.io/v2/cards"

    def __init__(self) -> None:
        self.api_key = os.getenv("POKEMON_TCG_API_KEY", "").strip()
        self.session = requests.Session()
        self._cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key} if self.api_key else {}

    def _request(self, params: dict[str, Any], timeout: tuple[int, int]) -> dict[str, Any]:
        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                headers=self.headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout as exc:
            raise RuntimeError("The Pokémon TCG catalog is responding too slowly. Please try again.") from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "The Pokémon TCG API could not be reached. Check your internet connection and try again."
            ) from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise RuntimeError(f"The Pokémon TCG API returned an error (HTTP {status}). Please try again.") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Card catalog request failed: {exc}") from exc

    @staticmethod
    def _parse_card(raw: dict[str, Any]) -> dict[str, Any]:
        card_set = raw.get("set", {})
        images = raw.get("images", {})
        printed_total = card_set.get("printedTotal") or card_set.get("total")
        raw_number = str(raw.get("number", ""))
        display_number = f"{raw_number}/{printed_total}" if printed_total else raw_number
        internal_set_id = str(card_set.get("id", ""))
        set_code = card_set.get("ptcgoCode") or internal_set_id
        set_code = SET_CODE_ALIASES.get(str(set_code).casefold(), set_code)

        supertype = str(raw.get("supertype", ""))
        subtypes = [str(value) for value in raw.get("subtypes", [])]
        card_types = [str(value) for value in raw.get("types", [])]
        rules = [str(value) for value in raw.get("rules", [])]
        rule_box = bool(rules) and any(
            marker in " ".join(rules).casefold()
            for marker in ("rule box", "pokémon ex", "pokémon-gx", "pokémon v", "radiant pokémon")
        )

        return {
            "id": raw["id"],
            "name": raw.get("name", "Unknown"),
            "number": display_number,
            "raw_number": raw_number,
            "printed_total": printed_total,
            "rarity": raw.get("rarity", ""),
            "set_name": card_set.get("name", "Unknown Set"),
            "set_code": set_code,
            "image_url": images.get("small") or images.get("large", ""),
            "supertype": supertype,
            "subtypes": subtypes,
            "card_category": normalize_card_category(supertype, subtypes),
            "pokemon_types": card_types,
            "hp": str(raw.get("hp", "")),
            "regulation_mark": str(raw.get("regulationMark", "")),
            "rule_box": rule_box,
            "is_promo": "promo" in {value.casefold() for value in subtypes} or "promo" in str(set_code).casefold(),
        }

    def search_cards(self, name: str, number: str = "") -> list[dict[str, Any]]:
        clean_name = name.strip()
        clean_number = number.strip().split("/")[0]
        cache_key = (clean_name.casefold(), clean_number.casefold())
        if cache_key in self._cache:
            return [card.copy() for card in self._cache[cache_key]]

        query_parts: list[str] = []
        if clean_name:
            query_parts.append(f'name:"{clean_name.replace(chr(34), "")}"')
        if clean_number:
            query_parts.append(f'number:"{clean_number}"')
        if not query_parts:
            return []

        payload = self._request(
            {
                "q": " ".join(query_parts),
                "pageSize": 50,
                "select": "id,name,number,rarity,set,supertype,subtypes,types,hp,rules,regulationMark",
            },
            timeout=(4, 12),
        )
        cards = [self._parse_card(raw) for raw in payload.get("data", [])]
        self._cache[cache_key] = [card.copy() for card in cards]
        return cards

    def fetch_card_page(self, page: int, page_size: int = 250) -> tuple[list[dict[str, Any]], int]:
        payload = self._request(
            {
                "page": page,
                "pageSize": page_size,
                "select": (
                    "id,name,number,rarity,set,images,supertype,subtypes,types,hp,rules,regulationMark"
                ),
            },
            timeout=(8, 60),
        )
        cards = [self._parse_card(raw) for raw in payload.get("data", [])]
        total_count = int(payload.get("totalCount") or len(cards))
        return cards, total_count
