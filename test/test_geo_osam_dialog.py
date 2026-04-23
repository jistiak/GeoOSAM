# coding=utf-8
"""Dialog test.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'geoosamplugin@gmail.com'
__date__ = '2025-06-26'
__copyright__ = 'Copyright 2025, Ofer Butbega'

import unittest

from qgis.PyQt.QtWidgets import QDialogButtonBox, QDialog

from geo_osam_dialog import SegSamDialog

from utilities import get_qgis_app
QGIS_APP = get_qgis_app()


def _qt_enum(owner, scoped_name, legacy_name=None):
    value = owner
    try:
        for part in scoped_name.split("."):
            value = getattr(value, part)
        return value
    except AttributeError:
        return getattr(owner, legacy_name or scoped_name.rsplit(".", 1)[-1])


BUTTON_OK = _qt_enum(QDialogButtonBox, "StandardButton.Ok", "Ok")
BUTTON_CANCEL = _qt_enum(QDialogButtonBox, "StandardButton.Cancel", "Cancel")
DIALOG_ACCEPTED = _qt_enum(QDialog, "DialogCode.Accepted", "Accepted")
DIALOG_REJECTED = _qt_enum(QDialog, "DialogCode.Rejected", "Rejected")


class SegSamDialogTest(unittest.TestCase):
    """Test dialog works."""

    def setUp(self):
        """Runs before each test."""
        self.dialog = SegSamDialog(None)

    def tearDown(self):
        """Runs after each test."""
        self.dialog = None

    def test_dialog_ok(self):
        """Test we can click OK."""

        button = self.dialog.button_box.button(BUTTON_OK)
        button.click()
        result = self.dialog.result()
        self.assertEqual(result, DIALOG_ACCEPTED)

    def test_dialog_cancel(self):
        """Test we can click cancel."""
        button = self.dialog.button_box.button(BUTTON_CANCEL)
        button.click()
        result = self.dialog.result()
        self.assertEqual(result, DIALOG_REJECTED)


if __name__ == "__main__":
    suite = unittest.makeSuite(SegSamDialogTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
