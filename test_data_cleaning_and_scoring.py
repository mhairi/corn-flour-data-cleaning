import pytest
import pandas as pd

from data_cleaning_and_scoring import (
    coalesce_columns,
    remove_duplicates,
    check_for_label,
    count_target_ingredients,
    count_target_products,
    is_target_industry,
    count_products_using_target,
    score_lead,
)


class TestDataCleaning:

    def test_coalesce(self):
        df = pd.DataFrame(
            {
                "keep": [1, 1, None, 3, None],
                "delete": [5, None, 2, None, None],
                "other": [1, 4, 5, 6, None],
            }
        )

        result = coalesce_columns(df, [("keep", "delete")])
        expected = pd.DataFrame(
            {"keep": [1.0, 1.0, 2.0, 3.0, None], "other": [1, 4, 5, 6, None]}
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_remove_duplicates_basic(self):
        """Test basic duplicate removal"""
        df = pd.DataFrame(
            {
                "company_name": ["A", "B", "A"],
                "keywords": ["a;b", "a;b", "b;c"],
            }
        )

        result = remove_duplicates(df)
        expected = pd.DataFrame(
            {"company_name": ["A", "B"], "keywords": ["a;b", "a;b"]}
        )

        pd.testing.assert_frame_equal(result, expected, check_index_type=False)


class TestCheckForLabel:

    def test_check_for_label_true(self):
        assert check_for_label("corn-starch; eco", "corn-starch") == True
        assert check_for_label("eco; corn-starch", "corn-starch") == True

    def test_check_for_label_false(self):
        assert check_for_label("dessert; eco", "corn-starch") == False

    def test_check_for_label_commas(self):
        assert check_for_label("corn-starch, eco", "corn-starch") == True


class TestIngredientsExtractionFunctions:

    @pytest.fixture
    def test_ingredients(self):
        return [
            {"name": "cake", "ingredients": ["flour", "sugar", "eggs"]},
            {"name": "biscuits", "ingredients": ["flour", "butter", "sugar"]},
            {"name": "face powder"},
            {"name": "cream", "ingredients": ["milk", "fat", "emulsifier"]},
        ]

    def test_count_target_ingredients(self, test_ingredients):
        result = count_target_ingredients(test_ingredients, ["flour", "milk"])
        assert result == 3

    def test_count_target_products(self, test_ingredients):
        result = count_target_products(
            test_ingredients, ["cream", "face powder", "cheese"]
        )
        assert result == 2


class TestInfoExtractionFunctions:

    @pytest.fixture
    def test_info(self):
        return "We produce bakery. Our products use thickener. Our products use binder."

    def test_is_target_industry(self, test_info):
        assert is_target_industry(test_info, ["bakery", "pet food"]) == True
        assert is_target_industry(test_info, ["pet food"]) == False

    def test_count_products_using_target(self, test_info):
        assert count_products_using_target(test_info, ["thickener", "binder"]) == 2
        assert count_products_using_target(test_info, ["starch"]) == 0


class TestLeadScoring:
    def test_score_lead(self):
        result = score_lead(
            has_corn_starch_label=True,
            n_target_ingredients=5,
            n_target_products=3,
            is_target_industry=True,
            n_products_using_target=2,
        )
        assert result == 170


if __name__ == "__main__":
    pytest.main([__file__])
