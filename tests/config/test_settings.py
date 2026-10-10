import unittest
from unittest.mock import patch, mock_open
from src.config.settings import Settings

class TestSettings(unittest.TestCase):
    """Tests the Settings class functionality."""

    def setUp(self):
        # Reset settings or ensure a clean state before each test
        self.default_settings = Settings()

    def test_default_settings_values(self):
        """Test that default configuration values are correctly set."""
        self.assertEqual(self.default_settings.k, 5)
        self.assertEqual(self.default_settings.chunking_strategy, "recursive")
        self.assertTrue(self.default_settings.use_reranking)
        self.assertIsInstance(self.default_settings.data_path, str)

    def test_from_yaml_success(self):
        """Test loading settings successfully from a valid YAML file."""
        yaml_content = """
        collection_name: "test_collection"
        chunk_size: 500
        use_reranking: false
        """

        # Mock the file reading
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            try:
                settings = Settings.from_yaml("dummy_path.yaml")
                self.assertEqual(settings.collection_name, "test_collection")
                self.assertEqual(settings.chunk_size, 500)
                self.assertFalse(settings.use_reranking)
            except Exception as e:
                self.fail(f"from_yaml raised an unexpected exception: {e}")

    def test_from_yaml_file_not_found(self):
        """Test that FileNotFoundError is raised when YAML file does not exist."""
        with self.assertRaises(FileNotFoundError):
            Settings.from_yaml("non_existent_file.yaml")

    def test_from_yaml_yaml_import_error(self):
        """Test that ImportError is raised if PyYAML is not installed."""
        # Temporarily mock yaml to be None to simulate missing dependency
        from unittest.mock import patch
        with patch('src.config.settings.yaml', None):
            with self.assertRaises(ImportError) as cm:
                Settings.from_yaml("dummy_path.yaml")
            self.assertIn("PyYAML is not installed", str(cm.exception))

if __name__ == '__main__':
    unittest.main()