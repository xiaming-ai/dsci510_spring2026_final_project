import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_access import fetch_acs_subject_group

class TestAPIAccess(unittest.TestCase):

    @patch('api_access.requests.get')
    def test_fetch_acs_subject_group_success(self, mock_get):
        # Mocking the response of the API
        mock_response = MagicMock()
        mock_response.json.return_value = [
            ["NAME", "S1901_C01_001E"],
            ["United States", "120000000"]
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        df = fetch_acs_subject_group(
            year=2024,
            survey="acs1",
            group="S1901",
            ucgid="0100000US",
            api_key="test_key"
        )

        # Assert requests.get was called correctly
        mock_get.assert_called_once_with(
            "https://api.census.gov/data/2024/acs/acs1/subject",
            params={
                "get": "group(S1901)",
                "ucgid": "0100000US",
                "key": "test_key"
            },
            timeout=30
        )

        # Assert the DataFrame is constructed correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), ["NAME", "S1901_C01_001E"])
        self.assertEqual(df.iloc[0]["NAME"], "United States")
        self.assertEqual(df.iloc[0]["S1901_C01_001E"], "120000000")

    @patch('api_access.requests.get')
    def test_fetch_acs_subject_group_invalid_format(self, mock_get):
        # Mocking a bad JSON response (e.g. error message returned as dict)
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "unknown issue"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            fetch_acs_subject_group()
        
        self.assertIn("Unexpected API response format", str(context.exception))

if __name__ == '__main__':
    unittest.main()
