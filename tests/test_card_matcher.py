from __future__ import annotations

import unittest
from typing import Any

from src.card_matcher import CardMatcher


class FakeCatalog:
    def __init__(self, cards: list[dict[str, Any]]) -> None:
        self.cards = cards

    def search_cards(
        self,
        name: str = "",
        number: str = "",
        set_query: str = "",
        printed_total: int | None = None,
        limit: int = 100,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        rows = list(self.cards)
        if name:
            rows = [card for card in rows if name.casefold() in card["name"].casefold()]
        if number:
            expected = number.split("/", 1)[0].lstrip("0") or "0"
            rows = [
                card
                for card in rows
                if (str(card["raw_number"]).lstrip("0") or "0") == expected
            ]
        if set_query:
            query = set_query.casefold()
            rows = [
                card
                for card in rows
                if query in card["set_code"].casefold()
                or query in card["set_name"].casefold()
            ]
        if printed_total is not None:
            rows = [card for card in rows if card["printed_total"] == printed_total]
        return rows[:limit]


def card(
    name: str,
    set_code: str,
    number: str,
    total: int,
    regulation: str = "",
) -> dict[str, Any]:
    return {
        "id": f"{set_code}-{number}",
        "name": name,
        "set_name": set_code,
        "set_code": set_code,
        "raw_number": number,
        "number": f"{number}/{total}",
        "printed_total": total,
        "regulation_mark": regulation,
    }


class CardMatcherResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cards = [
            card("Slowpoke", "SCR", "57", 142, "H"),
            card("Slowpoke", "PRE", "18", 131, "H"),
            card("Slowpoke", "MEW", "79", 165, "G"),
            card("Genesect ex", "BLK", "67", 86, "I"),
            card("Hydreigon ex", "WHT", "67", 86, "I"),
            card("Sliggoo", "CRI", "67", 86, "J"),
            card("Academy at Night", "SFA", "54", 64, "H"),
            card("Mega Kangaskhan ex", "MEG", "104", 132, "I"),
            card("Celebi", "MEG", "12", 132, "I"),
            card("Celebi", "VIV", "9", 185, "D"),
        ]
        self.matcher = CardMatcher(FakeCatalog(self.cards))

    def test_exact_title_and_regulation_keep_ambiguous_prints_for_review(self) -> None:
        result = self.matcher.resolve(
            name="Slowpoke",
            set_query="SCR",
            fuzzy_name_hint="Slowpoke",
            prefer_name=True,
            trust_set_hint=False,
            regulation_mark="H",
        )
        self.assertFalse(result.decisive)
        self.assertCountEqual(
            ["SCR", "PRE"], [item["set_code"] for item in result.matches]
        )

    def test_exact_fraction_recovers_missing_ex_suffix(self) -> None:
        result = self.matcher.resolve(
            name="Genesect",
            set_query="BKJ",
            number_text="67/86",
            fuzzy_name_hint="Genesect",
            prefer_name=True,
        )
        self.assertTrue(result.decisive)
        self.assertEqual("Genesect ex", result.name_value)
        self.assertEqual("BLK", result.set_value)

    def test_leading_number_noise_is_corrected_only_with_separated_title(self) -> None:
        result = self.matcher.resolve(
            name="Genesectex",
            set_query="BUK",
            number_text="267/86",
            fuzzy_name_hint="Genesectex",
            prefer_name=True,
        )
        self.assertTrue(result.decisive)
        self.assertEqual("67/86", result.number_value)
        self.assertEqual("leading_number_correction", result.strategy)

    def test_weak_title_does_not_force_leading_number_correction(self) -> None:
        result = self.matcher.resolve(
            name="DASG",
            set_query="BLK",
            number_text="767/86",
            fuzzy_name_hint="DASG",
            prefer_name=True,
        )
        self.assertFalse(result.decisive)

    def test_unique_exact_title_survives_wrong_identifier_clues(self) -> None:
        result = self.matcher.resolve(
            name="Academy at Night",
            set_query="SGA",
            number_text="34/64",
            fuzzy_name_hint="Academy at Night",
            prefer_name=True,
            trust_set_hint=False,
        )
        self.assertTrue(result.decisive)
        self.assertEqual("SFA", result.set_value)
        self.assertEqual("54/64", result.number_value)

    def test_alternate_title_resolves_ex_card_with_exact_fraction(self) -> None:
        result = self.matcher.resolve(
            name="Megal",
            set_query="JMG",
            number_text="104/132",
            fuzzy_name_hint="Megal",
            title_hints=(("Megal", 95.0), ("MegaKangashmex", 38.0)),
            prefer_name=True,
        )
        self.assertTrue(result.decisive)
        self.assertEqual("Mega Kangaskhan ex", result.name_value)

    def test_base_and_ex_title_variants_require_review_without_identifier(self) -> None:
        cards = [
            card("Genesect", "PLB", "10", 101),
            card("Genesect ex", "BLK", "67", 86, "I"),
        ]
        matcher = CardMatcher(FakeCatalog(cards))
        result = matcher.resolve(
            name="Genesect",
            fuzzy_name_hint="Genesect",
            title_hints=(("Genesect", 90.0), ("Genesect ex", 55.0)),
            prefer_name=True,
        )
        self.assertFalse(result.decisive)
        self.assertCountEqual(
            ["Genesect", "Genesect ex"],
            [item["name"] for item in result.matches],
        )

    def test_lower_confidence_exact_title_replaces_selected_junk_for_review(self) -> None:
        result = self.matcher.resolve(
            name="DASIS",
            set_query="MEG",
            fuzzy_name_hint="DASIS",
            title_hints=(("DASIS", 79.0), ("Celebi", 64.0)),
            prefer_name=True,
            trust_set_hint=False,
        )
        self.assertFalse(result.decisive)
        self.assertEqual("Celebi", result.name_value)
        self.assertTrue(result.matches)
        self.assertTrue(all(item["name"] == "Celebi" for item in result.matches))


if __name__ == "__main__":
    unittest.main()
