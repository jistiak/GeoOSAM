import os

from import_utils import ensure_import_path


def test_ensure_import_path_prepends_new_path(tmp_path):
    search_path = [os.fspath(tmp_path / 'other')]
    plugin_path = tmp_path / 'GeoOSAM'

    ensure_import_path(plugin_path, prepend=True, search_path=search_path)

    assert search_path == [os.fspath(plugin_path), os.fspath(tmp_path / 'other')]


def test_ensure_import_path_moves_existing_path_without_duplicates(tmp_path):
    plugin_path = os.fspath(tmp_path / 'GeoOSAM')
    equivalent_path = os.path.join(plugin_path, '.', '')
    search_path = ['first', plugin_path, equivalent_path]

    ensure_import_path(plugin_path, prepend=True, search_path=search_path)

    assert search_path == [plugin_path, 'first']
