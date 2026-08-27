from class_catalog import build_class_catalog, order_class_names, parse_class_list_text
from class_matching import get_class_family


DEFAULTS = {
    'Buildings': {
        'color': '220,20,60',
        'description': 'Structures',
        'batch_defaults': {'min_size': 150, 'max_objects': 20},
    },
    'Water': {
        'color': '30,144,255',
        'description': 'Water bodies',
        'batch_defaults': {'min_size': 250, 'max_objects': 8},
    },
}


def test_parse_class_list_preserves_order_and_removes_duplicates():
    text = "class\nTrees,Building roof\nWater\ntrees\n"

    assert parse_class_list_text(text) == ['Trees', 'Building roof', 'Water']


def test_parse_class_list_respects_csv_quoting():
    assert parse_class_list_text('"Road, primary"\nBuildings') == [
        'Road, primary', 'Buildings']


def test_catalogue_keeps_defaults_and_uses_generic_custom_settings():
    catalogue = build_class_catalog(
        ['Water', 'Building roof', 'Unknown object'],
        DEFAULTS,
        ['1,2,3', '4,5,6'])

    assert list(catalogue) == ['Water', 'Building roof', 'Unknown object']
    assert catalogue['Water']['batch_defaults'] == {
        'min_size': 250, 'max_objects': 8}
    assert catalogue['Water']['source'] == 'default'
    assert catalogue['Building roof']['helper_family'] == 'building'
    assert catalogue['Building roof']['batch_defaults'] == {
        'min_size': 50, 'max_objects': 25}
    assert catalogue['Unknown object']['helper_family'] == 'general'


def test_similar_helper_matching_is_conservative():
    assert get_class_family('Building roof') == 'building'
    assert get_class_family('Tree labels') == 'vegetation'
    assert get_class_family('Water road') == 'general'
    assert get_class_family('Unrelated object') == 'general'


def test_quick_table_sorting_keeps_a_restorable_source_order():
    source = ['Water', 'Tree', 'Buildings']

    assert order_class_names(source, source, 'source') == source
    assert order_class_names(source, source, 'ascending') == [
        'Buildings', 'Tree', 'Water']
    assert order_class_names(source, source, 'descending') == [
        'Water', 'Tree', 'Buildings']
