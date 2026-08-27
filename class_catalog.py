"""Utilities for loading and building GeoOSAM class catalogues.

This module intentionally has no QGIS or Qt dependencies so class-file parsing
and helper selection can be tested outside a running QGIS session.
"""

import copy
import csv
import io

try:
    from .class_matching import get_class_family, normalize_class_name
except ImportError:
    from class_matching import get_class_family, normalize_class_name


GENERIC_BATCH_DEFAULTS = {'min_size': 50, 'max_objects': 25}
HEADER_NAMES = {'class', 'classes', 'class name', 'class_name', 'label', 'labels'}


def parse_class_list_text(text):
    """Parse comma- or line-separated class names while preserving order.

    CSV quoting is respected, blank values are ignored, and duplicate names are
    removed case-insensitively after normalisation. A conventional first-cell
    header such as ``class`` or ``label`` is ignored.
    """
    names = []
    seen = set()

    for row in csv.reader(io.StringIO(text or '')):
        for value in row:
            class_name = value.strip()
            normalized = normalize_class_name(class_name)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            names.append(class_name)

    if len(names) > 1 and normalize_class_name(names[0]) in HEADER_NAMES:
        names.pop(0)

    return names


def build_class_catalog(class_names, default_classes, extra_colors):
    """Build an ordered class catalogue from imported class names.

    Exact built-in matches inherit their complete configuration. Other labels
    receive generic settings and a deterministic colour. Helper dispatch is
    still based on the label itself, so aliases or similar names can reuse a
    specialised helper while unrelated labels fall back to the general helper.
    """
    defaults_by_name = {
        normalize_class_name(name): (name, info)
        for name, info in default_classes.items()
    }
    palette = list(extra_colors) or ['128,128,128']
    catalogue = {}
    custom_index = 0

    for class_name in class_names:
        normalized = normalize_class_name(class_name)
        if normalized in defaults_by_name:
            _, default_info = defaults_by_name[normalized]
            catalogue[class_name] = copy.deepcopy(default_info)
            catalogue[class_name]['source'] = 'default'
            catalogue[class_name]['helper_family'] = get_class_family(class_name)
            continue

        helper_family = get_class_family(class_name)
        description = 'Custom class'
        if helper_family != 'general':
            description += f' · {helper_family} helper'

        catalogue[class_name] = {
            'color': palette[custom_index % len(palette)],
            'description': description,
            'batch_defaults': dict(GENERIC_BATCH_DEFAULTS),
            'source': 'custom',
            'helper_family': helper_family,
        }
        custom_index += 1

    return catalogue


def order_class_names(source_order, available_names, sort_mode='source'):
    """Order class names for the quick table without losing source order."""
    available = list(available_names)
    ordered = [name for name in source_order if name in available]
    ordered.extend(name for name in available if name not in ordered)

    if sort_mode == 'ascending':
        return sorted(ordered, key=str.casefold)
    if sort_mode == 'descending':
        return sorted(ordered, key=str.casefold, reverse=True)
    return ordered
