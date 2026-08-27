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
    for family, class_names in CLASS_FAMILIES.items():
        if normalized in class_names:
            return family

    # Allow reliable similar labels such as "building roof" or "tree labels"
    # without guessing when tokens point to more than one helper family.
    class_tokens = _class_tokens(normalized)
    if not class_tokens:
        return 'general'

    candidates = []
    for family, class_names in CLASS_FAMILIES.items():
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
