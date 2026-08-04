import unittest

from localforge_i18n import (
    LANGUAGE_CODES,
    LANGUAGE_FONTS,
    LANGUAGE_NAMES,
    TRANSLATIONS,
    translate,
)


class I18nTest(unittest.TestCase):
    def test_catalogs_have_the_same_keys(self):
        expected = set(TRANSLATIONS["th"])
        for language, catalog in TRANSLATIONS.items():
            self.assertEqual(set(catalog), expected, language)

    def test_all_languages_have_names_fonts_and_reverse_mapping(self):
        self.assertEqual(set(LANGUAGE_NAMES), set(TRANSLATIONS))
        self.assertEqual(set(LANGUAGE_FONTS), set(TRANSLATIONS))
        for code, name in LANGUAGE_NAMES.items():
            self.assertEqual(LANGUAGE_CODES[name], code)

    def test_format_values(self):
        self.assertEqual(translate("en", "copy_code", index=2), "Copy code 2")
        self.assertIn("/models", translate("ja", "models_location", path="/models"))

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(translate("missing", "send"), translate("en", "send"))


if __name__ == "__main__":
    unittest.main()
