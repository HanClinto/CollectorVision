import unittest

import collector_vision as cvg
from collector_vision.model_registry import available_models, get_model


class ModelRegistryTests(unittest.TestCase):
    def test_family_alias_resolves_to_current_default(self) -> None:
        model = get_model("cornelius")

        self.assertEqual(model.id, "cornelius-2.12")
        self.assertEqual(model.family, "cornelius")
        self.assertEqual(model.task, "corner-detection")

    def test_exact_model_id_remains_available(self) -> None:
        model = get_model("milo-1.0.0")

        self.assertEqual(model.version, "1.0.0")
        self.assertEqual(model.architecture, "mobilevit-xxs-arcface")

    def test_aliases_and_exact_ids_are_listed(self) -> None:
        self.assertEqual(
            available_models(),
            ("cornelius", "cornelius-2.12", "milo", "milo-1.0.0"),
        )

    def test_unknown_model_lists_supported_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "cornelius-2.12"):
            get_model("corndog")

    def test_registry_is_available_from_package_root(self) -> None:
        self.assertEqual(cvg.get_model("milo").id, "milo-1.0.0")
        self.assertIn("cornelius", cvg.available_models())


if __name__ == "__main__":
    unittest.main()
