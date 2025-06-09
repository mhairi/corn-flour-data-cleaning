import pandas as pd
import re
from typing import Any


# ---------------------------------------------------------------------------- #
#                                 Data Cleaning                                #
# ---------------------------------------------------------------------------- #


def load_json(file_path: str) -> pd.DataFrame:
    return pd.read_json(file_path, orient="records")


def coalesce_columns(
    df: pd.DataFrame, column_replacements: list[tuple[str, str]]
) -> pd.DataFrame:
    """Fix columns so that columns containing the same information are merged into one.
    Keeps the first column, merging in values in the second column if the first column has missing data.
    """
    df = df.copy()
    for column_keep, column_delete in column_replacements:

        df[column_keep] = df[column_keep].combine_first(df[column_delete])
        df = df.drop(columns=column_delete)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove companies with the same name as an existing company"""
    # Sort by 'keywords' first, so duplicates with missing keywords are dropped
    df = df.copy()
    df = df.sort_values(["company_name", "keywords"]).drop_duplicates(
        subset="company_name", keep="first", ignore_index=True
    )
    return df


# ---------------------------------------------------------------------------- #
#                                    Scoring                                   #
# ---------------------------------------------------------------------------- #


def check_for_label(labels: Any, label_to_check_for: str) -> bool:
    """Check if label list contains specified label"""
    if not isinstance(labels, str):
        return False
    labels_list = re.split(r"[;,]", labels)
    labels_list = [l.strip() for l in labels_list]
    if label_to_check_for in labels_list:
        return True
    else:
        return False


def count_target_ingredients(items: Any, target_ingredients: list[str]) -> int:
    """Count how often target ingredients are in items list."""
    if not isinstance(items, list):
        return 0

    count = 0
    for item in items:
        if "ingredients" in item.keys():
            count += sum(
                [
                    1
                    for ingredient in item["ingredients"]
                    if ingredient in target_ingredients
                ]
            )
    return count


def count_target_products(items: Any, target_products: list[str]) -> int:
    """Count how often target products are in items list."""
    if not isinstance(items, list):
        return 0

    return sum([1 for item in items if item["name"] in target_products])


def is_target_industry(info: Any, target_industries: list[str]) -> bool:
    """Checks if company is described as being one of the target industries"""
    if not isinstance(info, str):
        return False

    target_industries = [re.escape(industry) for industry in target_industries]
    regex = r"We produce (" + "|".join(target_industries) + ")"

    if re.match(regex, info):
        return True
    else:
        return False


def count_products_using_target(info: Any, target_uses: list[str]) -> int:
    """Count how often products are described as using one of target items"""
    if not isinstance(info, str):
        return 0
    target_uses = [re.escape(use) for use in target_uses]
    regex = r"Our products use (" + "|".join(target_uses) + ")"
    return len(re.findall(regex, info))


def score_lead(
    has_corn_starch_label: bool,
    n_target_ingredients: int,
    n_target_products: int,
    is_target_industry: bool,
    n_products_using_target: int,
) -> int:
    """Give relavance score to a lead, based on generated columns."""
    return (
        50 * has_corn_starch_label
        + 20 * is_target_industry
        + 10 * n_target_ingredients
        + 10 * n_target_products
        + 10 * n_products_using_target
    )


if __name__ == "__main__":

    # Manually edited JSON file to fix incorrect JSON structure

    df = load_json("data/processed_buyer_leads.json")
    df = coalesce_columns(
        df,
        [
            ("company_name", "name"),
            ("company_name", "company"),
            ("info", "description"),
            ("info", "desc"),
            ("labels", "tags"),
            ("products", "product_list"),
            ("url", "website"),
            ("url", "site"),
        ],
    )
    df = remove_duplicates(df)

    df["has_corn_starch_label"] = df["labels"].apply(
        check_for_label, args=("corn-starch",)
    )

    possible_replaced_with_corn_starch = [
        "emulsifier",
        "flour",
        "gelatin",
        "mica",
        "talc",
        "zinc oxide",
    ]
    df["n_target_ingredients"] = df["items"].apply(
        count_target_ingredients, args=(possible_replaced_with_corn_starch,)
    )

    possible_corn_starch_product = [
        "biscuits",
        "cake",
        "chips",
        "face powder",
        "pudding",
    ]
    df["n_target_products"] = df["items"].apply(
        count_target_products, args=(possible_corn_starch_product,)
    )

    corn_starch_using_industries = [
        "snacks",
        "natural snacks",
        "bakery",
        "sauces",
        "cosmetics",
        "pet food",
    ]
    df["is_target_industry"] = df["info"].apply(
        is_target_industry, args=(corn_starch_using_industries,)
    )

    products_might_use_corn_starch = ["binder", "thickener", "maize powder", "starch"]
    df["n_products_using_target"] = df["info"].apply(
        count_products_using_target, args=(products_might_use_corn_starch,)
    )

    df["score"] = df.apply(
        lambda x: score_lead(
            x["has_corn_starch_label"],
            x["n_target_ingredients"],
            x["n_target_products"],
            x["is_target_industry"],
            x["n_products_using_target"],
        ),
        axis=1,
    )

    df.sort_values(["score"], ascending=False).to_csv(
        "data/cleaned_and_scored_buyer_leads.csv", index=False
    )
