"""Lightweight class-name normalisation and helper-family matching."""

import re


# Normalized, case-folded aliases. Related dataset labels can share a helper
# without forcing every imported class to match the built-in catalogue exactly.
CLASS_FAMILIES = {
    'vegetation': {
        'vegetation', 'grass', 'greenfield', 'tree', 'trees', 'tree canopy',
        'canopy', 'artificial turf'
    },
    'building': {
        'buildings', 'building', 'industrial', 'glass roof', 'green roof',
        'red roof', 'dark roof', 'industrial roof', 'pv', 'thermo', 'window',
        'solar tube'
    },
    'residential': {'residential'},
    'vehicle': {'vehicle', 'cars'},
    'vessel': {'vessels', 'vessel', 'ship', 'ships', 'boat', 'boats'},
    'water': {'water'},
    'agriculture': {'agriculture', 'field'},
    'road': {'road', 'roads', 'railway', 'bike lane'},
}


# Background-only German aliases for imported class files. These are helper
# hints, not additional visible defaults: German labels keep their original
# spelling in the UI and still receive generic custom-class settings.
GERMAN_CLASS_FAMILIES = {
    'vegetation': {
        'bewuchs', 'gras', 'gruenflaeche', 'grünfläche', 'gruenland',
        'grünland', 'kunstrasen', 'rasen', 'wiese', 'baum', 'baeume',
        'bäume', 'baumbestand', 'baumkrone', 'baumkronen', 'kronendach'
    },
    'building': {
        'gebaeude', 'gebäude', 'bauwerk', 'bauwerke', 'haus', 'haeuser',
        'häuser', 'industrie', 'industriegebaeude', 'industriegebäude',
        'gebaeudedach', 'gebäudedach', 'dach', 'daecher', 'dächer',
        'glasdach', 'glasdaecher', 'glasdächer', 'gruendach', 'gründach',
        'gruendaecher', 'gründächer', 'dachbegruenung', 'dachbegrünung',
        'rotes dach', 'rote daecher', 'rote dächer', 'dunkles dach',
        'dunkle daecher', 'dunkle dächer', 'industriedach',
        'industriedaecher', 'industriedächer', 'photovoltaik',
        'photovoltaikanlage', 'photovoltaikanlagen', 'solarmodul',
        'solarmodule', 'solarthermie', 'solarthermiekollektor',
        'solarthermiekollektoren', 'fenster', 'lichttunnel',
        'tageslichtrohr', 'tageslichtroehre', 'tageslichtröhre'
    },
    'residential': {
        'wohnbebauung', 'wohngebaeude', 'wohngebäude', 'wohngebiet',
        'wohngebiete', 'wohnsiedlung', 'wohnsiedlungen'
    },
    'vehicle': {
        'fahrzeug', 'fahrzeuge', 'auto', 'autos', 'pkw', 'lkw', 'bus', 'busse'
    },
    'vessel': {
        'wasserfahrzeug', 'wasserfahrzeuge', 'schiff', 'schiffe', 'boot', 'boote'
    },
    'water': {
        'wasser', 'gewaesser', 'gewässer', 'fluss', 'fluesse', 'flüsse',
        'see', 'seen', 'teich', 'teiche'
    },
    'agriculture': {
        'landwirtschaft', 'ackerbau', 'acker', 'aecker', 'äcker', 'ackerland',
        'agrarflaeche', 'agrarfläche', 'agrarflaechen', 'agrarflächen',
        'feld', 'felder'
    },
    'road': {
        'strasse', 'strassen', 'fahrbahn', 'fahrbahnen', 'weg', 'wege',
        'eisenbahn', 'bahnstrecke', 'bahnstrecken', 'gleis', 'gleise',
        'radweg', 'radwege', 'fahrradweg', 'fahrradwege'
    },
}


def _family_aliases(family):
    """Return English and background German aliases for a helper family."""
    return CLASS_FAMILIES.get(family, set()) | GERMAN_CLASS_FAMILIES.get(
        family, set())


def normalize_class_name(class_name):
    """Case-fold and collapse punctuation/whitespace for class matching."""
    normalized = re.sub(r"[^\w]+", " ", (class_name or "").strip().casefold())
    return " ".join(normalized.split())


def _class_tokens(class_name):
    """Return lightly singularised tokens for conservative similar matching."""
    tokens = []
    for token in normalize_class_name(class_name).split():
        if token.endswith('ies') and len(token) > 4:
            token = token[:-3] + 'y'
        elif token.endswith('s') and not token.endswith('ss') and len(token) > 3:
            token = token[:-1]
        tokens.append(token)
    return frozenset(tokens)


def get_class_family(class_name):
    """Return a specialised helper family or ``general`` for an unknown name."""
    normalized = normalize_class_name(class_name)
    for family in CLASS_FAMILIES:
        class_names = _family_aliases(family)
        if normalized in class_names:
            return family

    # Allow reliable similar labels such as "building roof" or "tree labels"
    # without guessing when tokens point to more than one helper family.
    class_tokens = _class_tokens(normalized)
    if not class_tokens:
        return 'general'

    candidates = []
    for family in CLASS_FAMILIES:
        class_names = _family_aliases(family)
        best_score = 0
        for alias in class_names:
            alias_tokens = _class_tokens(alias)
            if alias_tokens and (
                    alias_tokens.issubset(class_tokens)
                    or class_tokens.issubset(alias_tokens)):
                best_score = max(best_score, len(alias_tokens & class_tokens))
        if best_score:
            candidates.append((best_score, family))

    if candidates:
        top_score = max(score for score, _ in candidates)
        top_families = {
            family for score, family in candidates if score == top_score
        }
        if len(top_families) == 1:
            return top_families.pop()
    return 'general'


def class_uses_helper(class_name, *families):
    """Check whether a class belongs to one of the requested helper families."""
    return get_class_family(class_name) in families
