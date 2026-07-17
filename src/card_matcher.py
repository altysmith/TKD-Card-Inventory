from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol


class CardCatalog(Protocol):
    def search_cards(self, **kwargs: Any) -> list[dict[str, Any]]: ...


@dataclass
class MatchResolution:
    matches: list[dict[str, Any]] = field(default_factory=list)
    decisive: bool = False
    name_value: str | None = None
    set_value: str | None = None
    number_value: str | None = None
    message: str = ""
    strategy: str = "none"


class CardMatcher:
    """Resolve OCR and manual clues against the local catalog without Qt."""

    SEARCH_LIMIT = 100
    TITLE_SEARCH_LIMIT = 500
    REVIEW_LIMIT = 20

    EXACT_SIMILARITY = 1.0
    FRACTION_TITLE_NAME_MIN = 0.72
    FRACTION_TITLE_MARGIN_MIN = 0.15
    FRACTION_TITLE_SET_MIN = 0.60
    FRACTION_TITLE_STRONG_NAME_MIN = 0.90
    TITLE_SET_MIN = 0.80
    TITLE_NAME_MIN = 0.62
    TITLE_MARGIN_MIN = 0.10
    CATALOG_TITLE_NAME_MIN = 0.45
    REVIEW_TITLE_NAME_MIN = 0.35
    RELAXED_SET_MIN = 0.60
    RELAXED_SET_MARGIN_MIN = 0.15
    NUMBER_FRAGMENT_MIN = 0.50
    NUMBER_FRAGMENT_MARGIN_MIN = 0.20
    NUMBER_FRAGMENT_MAX_CANDIDATES = 5
    TITLE_HINT_CONFIDENCE_MIN = 8.0

    SET_SCORE_WEIGHT = 25.0
    NAME_SCORE_WEIGHT = 60.0
    EXACT_NUMBER_SCORE = 12.0
    WRONG_NUMBER_SCORE = -2.0
    EXACT_TOTAL_SCORE = 5.0
    WRONG_TOTAL_SCORE = -1.0

    def __init__(self, catalog: CardCatalog) -> None:
        self.catalog = catalog

    @staticmethod
    def _normalize_name(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @staticmethod
    def set_code_similarity(set_hint: str, set_code: str) -> float:
        hint = re.sub(r"[^A-Z0-9]", "", set_hint.upper())
        code = re.sub(r"[^A-Z0-9]", "", set_code.upper())
        if not hint or not code:
            return 0.0
        return SequenceMatcher(None, hint, code).ratio()

    @classmethod
    def name_similarity(cls, name_hint: str, card_name: str) -> float:
        hint = cls._normalize_name(name_hint)
        name = cls._normalize_name(card_name)
        if not hint or not name:
            return 0.0
        if hint == name:
            return cls.EXACT_SIMILARITY
        return SequenceMatcher(None, hint, name).ratio()

    @classmethod
    def title_texts(
        cls,
        primary: str,
        title_hints: tuple[tuple[str, float], ...] = (),
    ) -> tuple[str, ...]:
        values: list[str] = []
        seen: set[str] = set()
        for text, confidence in ((primary, 100.0), *title_hints):
            normalized = cls._normalize_name(text)
            if (
                not normalized
                or normalized in seen
                or (text != primary and confidence < cls.TITLE_HINT_CONFIDENCE_MIN)
            ):
                continue
            seen.add(normalized)
            values.append(text)
        return tuple(values)

    @classmethod
    def best_name_similarity(
        cls, name_hints: tuple[str, ...], card_name: str
    ) -> float:
        return max(
            (cls.name_similarity(hint, card_name) for hint in name_hints),
            default=0.0,
        )

    @staticmethod
    def collector_leading_noise_suffix(
        collector_hint: str, printed_total: int | None
    ) -> str:
        digits = re.sub(r"\D", "", collector_hint.split("/", 1)[0])
        if printed_total is None or len(digits) != 3:
            return ""
        if int(digits) <= printed_total:
            return ""
        suffix = digits[1:].lstrip("0") or "0"
        return suffix if int(suffix) <= printed_total else ""

    @classmethod
    def catalog_candidate_score(
        cls,
        card: dict[str, Any],
        set_hint: str = "",
        name_hint: str = "",
        name_hints: tuple[str, ...] = (),
        collector_hint: str = "",
        printed_total: int | None = None,
    ) -> float:
        score = 0.0
        if set_hint:
            score += cls.set_code_similarity(
                set_hint, str(card.get("set_code", ""))
            ) * cls.SET_SCORE_WEIGHT
        combined_name_hints = cls.title_texts(name_hint) + tuple(name_hints)
        if combined_name_hints:
            score += cls.best_name_similarity(
                combined_name_hints, str(card.get("name", ""))
            ) * cls.NAME_SCORE_WEIGHT
        if collector_hint:
            expected = re.sub(r"\D", "", collector_hint.split("/", 1)[0])
            actual = re.sub(
                r"\D",
                "",
                str(card.get("raw_number", card.get("number", ""))).split("/", 1)[0],
            )
            score += (
                cls.EXACT_NUMBER_SCORE
                if expected and expected == actual
                else cls.WRONG_NUMBER_SCORE
            )
        if printed_total is not None:
            score += (
                cls.EXACT_TOTAL_SCORE
                if card.get("printed_total") == printed_total
                else cls.WRONG_TOTAL_SCORE
            )
        return score

    @classmethod
    def rank_catalog_candidates(
        cls,
        cards: list[dict[str, Any]],
        set_hint: str,
        name_hint: str = "",
        name_hints: tuple[str, ...] = (),
        collector_hint: str = "",
        printed_total: int | None = None,
    ) -> list[dict[str, Any]]:
        return sorted(
            cards,
            key=lambda card: (
                -cls.catalog_candidate_score(
                    card,
                    set_hint=set_hint,
                    name_hint=name_hint,
                    name_hints=name_hints,
                    collector_hint=collector_hint,
                    printed_total=printed_total,
                ),
                str(card.get("name", "")).casefold(),
                str(card.get("set_name", "")).casefold(),
            ),
        )

    @classmethod
    def decisive_fraction_title_match(
        cls,
        cards: list[dict[str, Any]],
        set_hint: str,
        name_hint: str,
        name_hints: tuple[str, ...] = (),
    ) -> bool:
        all_name_hints = cls.title_texts(name_hint) + tuple(name_hints)
        if not cards or not all_name_hints:
            return False
        best_name = cls.best_name_similarity(
            all_name_hints, str(cards[0].get("name", ""))
        )
        second_name = (
            cls.best_name_similarity(all_name_hints, str(cards[1].get("name", "")))
            if len(cards) > 1
            else 0.0
        )
        best_set = cls.set_code_similarity(set_hint, str(cards[0].get("set_code", "")))
        return (
            best_name >= cls.FRACTION_TITLE_NAME_MIN
            and best_name - second_name >= cls.FRACTION_TITLE_MARGIN_MIN
            and (
                best_set >= cls.FRACTION_TITLE_SET_MIN
                or best_name >= cls.FRACTION_TITLE_STRONG_NAME_MIN
            )
        )

    @classmethod
    def decisive_title_match(
        cls,
        cards: list[dict[str, Any]],
        set_hint: str,
        name_hint: str,
        name_hints: tuple[str, ...] = (),
    ) -> bool:
        all_name_hints = cls.title_texts(name_hint) + tuple(name_hints)
        if not cards or not set_hint or not all_name_hints:
            return False
        best_set = cls.set_code_similarity(set_hint, str(cards[0].get("set_code", "")))
        best_name = cls.best_name_similarity(
            all_name_hints, str(cards[0].get("name", ""))
        )
        second_name = (
            cls.best_name_similarity(all_name_hints, str(cards[1].get("name", "")))
            if len(cards) > 1
            else 0.0
        )
        return (
            best_set >= cls.TITLE_SET_MIN
            and best_name >= cls.TITLE_NAME_MIN
            and best_name - second_name >= cls.TITLE_MARGIN_MIN
        )

    @classmethod
    def decisive_catalog_match(
        cls,
        cards: list[dict[str, Any]],
        set_hint: str,
        name_hint: str,
        collector_hint: str = "",
        name_hints: tuple[str, ...] = (),
    ) -> bool:
        all_name_hints = cls.title_texts(name_hint) + tuple(name_hints)
        if cls.decisive_title_match(
            cards, set_hint, name_hint, name_hints=name_hints
        ):
            return True
        if not cards or not set_hint or not all_name_hints or not collector_hint:
            return False
        expected = re.sub(r"\D", "", collector_hint.split("/", 1)[0])
        best_actual = re.sub(
            r"\D",
            "",
            str(cards[0].get("raw_number", cards[0].get("number", ""))).split("/", 1)[0],
        )
        second_actual = (
            re.sub(
                r"\D",
                "",
                str(cards[1].get("raw_number", cards[1].get("number", ""))).split("/", 1)[0],
            )
            if len(cards) > 1
            else ""
        )
        return (
            cls.set_code_similarity(set_hint, str(cards[0].get("set_code", "")))
            >= cls.TITLE_SET_MIN
            and cls.best_name_similarity(
                all_name_hints, str(cards[0].get("name", ""))
            )
            >= cls.CATALOG_TITLE_NAME_MIN
            and bool(expected)
            and best_actual == expected
            and second_actual != expected
        )

    @classmethod
    def narrow_exact_name_candidates(
        cls,
        cards: list[dict[str, Any]],
        name_hint: str,
        set_hint: str = "",
        collector_hint: str = "",
        printed_total: int | None = None,
        trust_set_hint: bool = True,
        regulation_mark: str = "",
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        candidates = [
            card
            for card in cards
            if cls.name_similarity(name_hint, str(card.get("name", "")))
            == cls.EXACT_SIMILARITY
        ]
        if not candidates:
            return [], False, False

        used_exact_identifier = False
        if collector_hint and printed_total is not None:
            expected = re.sub(r"\D", "", collector_hint.split("/", 1)[0])
            identifier_matches = [
                card
                for card in candidates
                if re.sub(
                    r"\D",
                    "",
                    str(card.get("raw_number", card.get("number", ""))).split("/", 1)[0],
                )
                == expected
                and card.get("printed_total") == printed_total
            ]
            if identifier_matches:
                candidates = identifier_matches
                used_exact_identifier = True

        used_set_hint = False
        if set_hint and trust_set_hint:
            set_matches = [
                card
                for card in candidates
                if cls.set_code_similarity(set_hint, str(card.get("set_code", "")))
                == cls.EXACT_SIMILARITY
            ]
            if set_matches:
                candidates = set_matches
                used_set_hint = True

        if regulation_mark:
            regulation_matches = [
                card
                for card in candidates
                if str(card.get("regulation_mark", "")).upper()
                == regulation_mark.upper()
            ]
            if regulation_matches:
                candidates = regulation_matches

        return candidates, used_set_hint, used_exact_identifier

    @classmethod
    def number_fragment_score(
        cls, card: dict[str, Any], hints: tuple[tuple[str, float], ...]
    ) -> float:
        raw_number = re.sub(
            r"\D",
            "",
            str(card.get("raw_number", card.get("number", ""))).split("/", 1)[0],
        )
        printed_total = re.sub(r"\D", "", str(card.get("printed_total") or ""))
        variants: set[str] = set()
        if raw_number:
            variants.update({raw_number, raw_number.zfill(3)})
        if printed_total:
            variants.update({printed_total, printed_total.zfill(3)})
        if raw_number and printed_total:
            variants.update(
                {
                    raw_number + printed_total,
                    raw_number.zfill(3) + printed_total.zfill(3),
                }
            )
        evidence: list[float] = []
        for hint, confidence in hints:
            digits = re.sub(r"\D", "", hint)
            if len(digits) < 3 or not variants:
                continue
            similarity = max(
                SequenceMatcher(None, digits, variant).ratio() for variant in variants
            )
            length_weight = min(1.0, len(digits) / 4.0)
            confidence_weight = 0.65 + min(100.0, confidence) / 285.0
            evidence.append(similarity * length_weight * confidence_weight)
        return sum(sorted(evidence, reverse=True)[:2])

    @classmethod
    def rank_number_fragment_candidates(
        cls,
        cards: list[dict[str, Any]],
        hints: tuple[tuple[str, float], ...],
    ) -> list[dict[str, Any]]:
        return sorted(
            cards,
            key=lambda card: (
                -cls.number_fragment_score(card, hints),
                str(card.get("set_name", "")).casefold(),
                str(card.get("number", "")),
            ),
        )

    @classmethod
    def decisive_number_fragment_match(
        cls,
        cards: list[dict[str, Any]],
        hints: tuple[tuple[str, float], ...],
    ) -> bool:
        if not cards or len(cards) > cls.NUMBER_FRAGMENT_MAX_CANDIDATES:
            return False
        best = cls.number_fragment_score(cards[0], hints)
        second = cls.number_fragment_score(cards[1], hints) if len(cards) > 1 else 0.0
        return (
            best >= cls.NUMBER_FRAGMENT_MIN
            and best - second >= cls.NUMBER_FRAGMENT_MARGIN_MIN
        )

    @staticmethod
    def _resolved(
        card: dict[str, Any], message: str, strategy: str
    ) -> MatchResolution:
        return MatchResolution(
            matches=[card],
            decisive=True,
            name_value=str(card.get("name", "")),
            set_value=str(card.get("set_code", "")),
            number_value=str(card.get("number", "")),
            message=message,
            strategy=strategy,
        )

    def resolve(
        self,
        *,
        name: str = "",
        set_query: str = "",
        number_text: str = "",
        printed_total: int | None = None,
        fuzzy_name_hint: str = "",
        title_hints: tuple[tuple[str, float], ...] = (),
        prefer_name: bool = False,
        trust_set_hint: bool = True,
        regulation_mark: str = "",
        number_hints: tuple[tuple[str, float], ...] = (),
    ) -> MatchResolution:
        name = name.strip()
        set_query = set_query.strip()
        number_text = number_text.strip()
        number = number_text.split("/", 1)[0] if number_text else ""
        if printed_total is None and "/" in number_text:
            total_text = number_text.split("/", 1)[1].strip()
            if total_text.isdigit():
                printed_total = int(total_text)
        title_hint = fuzzy_name_hint or name
        all_title_hints = self.title_texts(title_hint, title_hints)

        if all_title_hints and number and printed_total is not None:
            candidates = self.catalog.search_cards(
                number=number,
                printed_total=printed_total,
                limit=self.SEARCH_LIMIT,
            )
            candidates = self.rank_catalog_candidates(
                candidates,
                set_query,
                name_hint=title_hint,
                name_hints=all_title_hints,
                collector_hint=number,
                printed_total=printed_total,
            )
            if self.decisive_fraction_title_match(
                candidates,
                set_query,
                title_hint,
                name_hints=all_title_hints,
            ):
                card = candidates[0]
                return self._resolved(
                    card,
                    f"Catalog resolved {card.get('name', title_hint)} "
                    f"{card.get('set_code', set_query)} {card.get('number', number)}",
                    "fraction_title",
                )

        catalog_title_hint = ""
        title_candidates: list[dict[str, Any]] = []
        exact_title_groups: list[tuple[str, list[dict[str, Any]]]] = []
        if all_title_hints and (prefer_name or number or set_query):
            for candidate_hint in all_title_hints:
                candidate_cards = self.catalog.search_cards(
                    name=candidate_hint, limit=self.TITLE_SEARCH_LIMIT
                )
                exact_cards = [
                    card
                    for card in candidate_cards
                    if self.name_similarity(
                        candidate_hint, str(card.get("name", ""))
                    )
                    == self.EXACT_SIMILARITY
                ]
                if exact_cards:
                    exact_title_groups.append((candidate_hint, exact_cards))

        exact_names = {
            self._normalize_name(str(card.get("name", "")))
            for _hint, cards in exact_title_groups
            for card in cards
        }
        if len(exact_names) > 1:
            combined: dict[str, dict[str, Any]] = {}
            for _hint, cards in exact_title_groups:
                for card in cards:
                    combined[str(card.get("id", id(card)))] = card
            matches = self.rank_number_fragment_candidates(
                list(combined.values()), number_hints
            )
            return MatchResolution(
                matches=matches,
                message="Multiple title variants found; confirmation required",
                strategy="title_variant_review",
            )
        if exact_title_groups:
            catalog_title_hint, title_candidates = exact_title_groups[0]

        if catalog_title_hint:
            exact_title_count = sum(
                self.name_similarity(catalog_title_hint, str(card.get("name", "")))
                == self.EXACT_SIMILARITY
                for card in title_candidates
            )
            title_candidates, used_set_hint, used_exact_identifier = (
                self.narrow_exact_name_candidates(
                    title_candidates,
                    catalog_title_hint,
                    set_hint=set_query,
                    collector_hint=number,
                    printed_total=printed_total,
                    trust_set_hint=trust_set_hint,
                    regulation_mark=regulation_mark,
                )
            )
            if title_candidates:
                matches = self.rank_number_fragment_candidates(
                    title_candidates, number_hints
                )
                fragments_are_decisive = self.decisive_number_fragment_match(
                    matches, number_hints
                )
                if fragments_are_decisive:
                    matches = matches[:1]
                decisive = len(matches) == 1 and (
                    exact_title_count == 1
                    or used_set_hint
                    or used_exact_identifier
                    or fragments_are_decisive
                )
                ignored_set_hint = bool(set_query and not used_set_hint)
                prefix = (
                    f"Ignored conflicting set guess {set_query} | "
                    if ignored_set_hint
                    else ""
                )
                if decisive:
                    card = matches[0]
                    result = self._resolved(
                        card,
                        prefix
                        + f"Catalog resolved {card.get('name', catalog_title_hint)} "
                        + f"{card.get('set_code', '')} {card.get('number', '')}",
                        "exact_title",
                    )
                    return result
                message = (
                    "One print suggested by supporting evidence; confirmation required"
                    if len(matches) == 1
                    else f"Exact title found; {len(matches)} prints need set or number confirmation"
                )
                return MatchResolution(
                    matches=matches,
                    name_value=catalog_title_hint,
                    set_value="" if ignored_set_hint else None,
                    message=prefix + message,
                    strategy="exact_title_review",
                )

        matches = self.catalog.search_cards(
            name=name,
            set_query=set_query,
            number=number,
            printed_total=printed_total,
            limit=self.SEARCH_LIMIT,
        )
        used_relaxed_identifier_match = False
        if not matches and set_query and number:
            matches = self.catalog.search_cards(
                name=name,
                number=number,
                printed_total=printed_total,
                limit=self.SEARCH_LIMIT,
            )
            matches = self.rank_catalog_candidates(matches, set_query)
            if (
                matches
                and self.set_code_similarity(
                    set_query, str(matches[0].get("set_code", ""))
                )
                < self.RELAXED_SET_MIN
            ):
                matches = []
            used_relaxed_identifier_match = bool(matches)

        if not matches and number:
            matches = self.catalog.search_cards(
                name=name, number=number, limit=self.SEARCH_LIMIT
            )
            if set_query:
                matches = self.rank_catalog_candidates(matches, set_query)
            used_relaxed_identifier_match = bool(matches)

        corrected_number = self.collector_leading_noise_suffix(number, printed_total)
        if not matches and corrected_number and all_title_hints:
            candidates = self.catalog.search_cards(
                number=corrected_number,
                printed_total=printed_total,
                limit=self.SEARCH_LIMIT,
            )
            candidates = self.rank_catalog_candidates(
                candidates,
                set_query,
                name_hint=title_hint,
                name_hints=all_title_hints,
                collector_hint=corrected_number,
                printed_total=printed_total,
            )
            if self.decisive_fraction_title_match(
                candidates,
                set_query,
                title_hint,
                name_hints=all_title_hints,
            ):
                card = candidates[0]
                return self._resolved(
                    card,
                    f"Catalog corrected leading number noise to "
                    f"{card.get('number', corrected_number)}",
                    "leading_number_correction",
                )
            if candidates:
                matches = candidates[: self.REVIEW_LIMIT]

        if not matches and all_title_hints and set_query:
            candidates = self.catalog.search_cards(
                set_query=set_query, limit=self.TITLE_SEARCH_LIMIT
            )
            candidates = self.rank_catalog_candidates(
                candidates,
                set_query,
                name_hint=title_hint,
                name_hints=all_title_hints,
                collector_hint=number,
                printed_total=printed_total,
            )
            if self.decisive_catalog_match(
                candidates,
                set_query,
                title_hint,
                collector_hint=number,
                name_hints=all_title_hints,
            ):
                card = candidates[0]
                return self._resolved(
                    card,
                    f"Catalog resolved {card.get('name', title_hint)} "
                    f"{card.get('set_code', set_query)} {card.get('number', number)}",
                    "fuzzy_title_set",
                )
            matches = [
                card
                for card in candidates[: self.REVIEW_LIMIT]
                if self.best_name_similarity(
                    all_title_hints, str(card.get("name", ""))
                )
                >= self.REVIEW_TITLE_NAME_MIN
            ]

        name_value: str | None = None
        set_value: str | None = None
        number_value: str | None = None
        message = ""
        if matches and len(matches) == 1 and number and not set_query:
            best_code = str(matches[0].get("set_code", ""))
            if best_code:
                set_value = best_code
                number_value = str(matches[0].get("number", number))
                message = f"Catalog identified set {best_code}"
        elif used_relaxed_identifier_match and set_query:
            best_code = str(matches[0].get("set_code", "")) if matches else ""
            best_score = self.set_code_similarity(set_query, best_code)
            second_score = (
                self.set_code_similarity(set_query, str(matches[1].get("set_code", "")))
                if len(matches) > 1
                else 0.0
            )
            if (
                best_code
                and best_score >= self.RELAXED_SET_MIN
                and best_score - second_score >= self.RELAXED_SET_MARGIN_MIN
            ):
                matches = [
                    card
                    for card in matches
                    if str(card.get("set_code", "")).casefold()
                    == best_code.casefold()
                ]
                set_value = best_code
                number_value = str(matches[0].get("number", number))
                message = f"Catalog resolved {best_code} {matches[0].get('number', number)}"

        return MatchResolution(
            matches=matches,
            decisive=False,
            name_value=name_value,
            set_value=set_value,
            number_value=number_value,
            message=message,
            strategy="catalog_review" if matches else "no_match",
        )
