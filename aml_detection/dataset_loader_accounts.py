'''Loading and cleaning of the IBM AML accounts reference file, and enrichment
of transactions with account attributes.
data which can be used 
    entity_type  : Corporation / Partnership / Sole Proprietorship / Individual.
                  A real bank knows whether a customer is a business or a
                  person, so this is legitimate at scoring time.
    bank_country : Parsed out of the bank name. Cross-border flow is a
                  recognised AML risk factor.
EXPECTED SIGNAL STRENGTH

Measured lift for entity_type and bank_country on the full file is close to
1.0, meaning these attributes barely move the laundering rate on their own.
They are included as weak context features; the real signal is expected to
come from transaction behaviour and network structure.

'''

from __future__ import annotations
 
from pathlib import Path
 
import pandas as pd
from loguru import logger
 
from aml_detection.config import RAW_DATA_DIR,INTERIM_DATA_DIR
MERGED_INTERIM_DIR = INTERIM_DATA_DIR / "merged"
INTERIM_FILE = "transactions_merged.parquet"


ACCOUNT_COLUMNS: tuple[str, ...] = (
    "Bank Name",
    "Bank ID",
    "Account Number",
    "Entity ID",
    "Entity Name",
)


VALID_ENTITY_TYPES: tuple[str, ...] = (
    "Corporation",
    "Partnership",
    "Sole Proprietorship",
    "Individual",
)

ENRICHMENT_COLUMNS: tuple[str, ...] = ("entity_type", "bank_country")
 
 
class AccountsError(Exception):
    """Base class for problems with the accounts reference file."""
 
 
class AccountsNotFoundError(AccountsError):
    """Raised when the accounts file is not on disk."""
 
 
class AccountsSchemaError(AccountsError):
    """Raised when the accounts file has unexpected columns."""
 
 
class DuplicateAccountKeyError(AccountsError):
    """Raised when account keys are not unique.
 
    This matters because a non-unique key turns the enrichment merge into a
    fan-out that silently multiplies transaction rows.
    """

# --------------------------------------------------------------------------
# loading data
# --------------------------------------------------------------------------
def load_raw_accounts(
    filename: str = "HI-Small_accounts.csv",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    
    """Read the accounts CSV from disk, unmodified apart from validation."""
    path = (data_dir or RAW_DATA_DIR) / filename
    logger.info(f"Reading {path}")
 
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise AccountsNotFoundError(
            f"Accounts file not found at {path}.\n"
            f"Download {filename} from Kaggle and place it in {path.parent}/"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise AccountsError(f"{path} is empty.") from exc
 
    missing = [column for column in ACCOUNT_COLUMNS if column not in df.columns]
    if missing:
        raise AccountsSchemaError(
            f"{path.name} is missing expected columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )
 
    logger.success(f"Read {len(df):,} accounts")
    return df


# --------------------------------------------------------------------------
# initial basic cleaning
# --------------------------------------------------------------------------

def _add_account_key(df: pd.DataFrame) -> pd.DataFrame:
    """Build the join key, matching the namespacing used on transactions.
    merging on <bank id>_<account number>
    
    """
    return df.assign(
        account_key=lambda d: d["Bank ID"].astype(str) + "_" + d["Account Number"].astype(str)
    )
 
 
def _parse_entity_type(df: pd.DataFrame) -> pd.DataFrame:
    """Extract entity type from the free-text Entity Name.
 
    "Corporation #33520" -> "Corporation". Values outside the known set are
    bucketed into "Other".
    """
    return df.assign(
        entity_type=lambda d: d["Entity Name"]
        .str.split("#")
        .str[0]
        .str.strip()
        .where(lambda s: s.isin(VALID_ENTITY_TYPES), "Other")
    )

def _parse_bank_country(df: pd.DataFrame) -> pd.DataFrame:
    """Extract country from the bank name where the naming pattern allows.
 
    "Portugal Bank #4507"          -> "Portugal"
    "National Bank of Harrisburg"  -> "Named" (no country encoded)
 
    A naive split on the word "Bank" produces junk categories like "National"
    and "Savings", so an anchored pattern is used instead.
    """
    return df.assign(
        bank_country=lambda d: d["Bank Name"]
        .str.extract(r"^(.+?)\s+Bank\s+#\d+$")[0]
        .fillna("Named")
        .str.strip()
    )


def _validate_unique_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Guard against a fan-out merge before it silently corrupts the data."""
    duplicated = df["account_key"].duplicated().sum()
    if duplicated:
        raise DuplicateAccountKeyError(
            f"{duplicated:,} duplicate account keys found. Merging on a "
            "non-unique key would multiply transaction rows. Deduplicate the "
            "accounts file before enrichment."
        )
    return df


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the join key and the attributes cleared for use.
 
    Entity ID, Account Number and the Entity Name suffix are dropped here
    rather than downstream, so identifiers cannot leak into features by
    accident later.
    """
    return df[["account_key", *ENRICHMENT_COLUMNS]]

# --------------------------------------------------------------------------
# cleaning data
# --------------------------------------------------------------------------

def clean_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every accounts cleaning step, in order."""
    logger.info("Cleaning accounts")
 
    cleaned = (
        df.pipe(_add_account_key)
        .pipe(_parse_entity_type)
        .pipe(_parse_bank_country)
        .pipe(_validate_unique_keys)
        .pipe(_select_columns)
    )
 
    logger.success(
        f"Cleaned {len(cleaned):,} accounts | "
        f"{cleaned['entity_type'].nunique()} entity types | "
        f"{cleaned['bank_country'].nunique()} bank countries"
    )
    return cleaned



def load_accounts(
    filename: str = "HI-Small_accounts.csv",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load and clean the accounts reference file in one call."""
    return clean_accounts(load_raw_accounts(filename=filename, data_dir=data_dir))

# --------------------------------------------------------------------------
# merging two dataframes
# --------------------------------------------------------------------------



def merge_accounts(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach account attributes to both sides of every transaction."""
    accounts = load_accounts() if accounts is None else accounts
    logger.info(f"Merging accounts onto {len(transactions):,} transactions")

    merged = transactions
    for side in ("src", "dst"):
        merged = merged.merge(
            accounts.rename(columns={c: f"{side}_{c}" for c in ENRICHMENT_COLUMNS}),
            left_on=f"{side}_account",
            right_on="account_key",
            how="left",
            validate="many_to_one",
        ).drop(columns="account_key")

    if len(merged) != len(transactions):
        raise AccountsError(
            f"Row count changed: {len(transactions):,} -> {len(merged):,}. Fan-out merge."
        )

    logger.success(
        f"Merged | unmatched src {merged['src_entity_type'].isna().mean():.2%} "
        f"| dst {merged['dst_entity_type'].isna().mean():.2%}"
    )
    return merged




def save_data_interim(df: pd.DataFrame, filename: str = INTERIM_FILE) -> Path:
    """Write merged frame to data/interim as parquet (preserves dtypes)."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DATA_DIR / filename
    df.to_parquet(path, index=False)
    logger.success(f"Wrote {len(df):,} rows to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path



def build_interim(transactions: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    """Load cleaned transactions, merge accounts, persist to interim."""
    if transactions is None:
        from aml_detection.dataset_loader import TRANSACTION_INTERIM_DIR

        path = TRANSACTION_INTERIM_DIR / "transaction_cleaned.parquet"
        if not path.exists():
            raise AccountsError(
                f"{path} not found. Run: python -m aml_detection.dataset_loader"
            )
        transactions = pd.read_parquet(path)
        logger.info(f"Loaded {len(transactions):,} cleaned transactions")

    merged = merge_accounts(transactions, load_accounts())
    if save:
        save_data_interim(merged)
    return merged




def save_data_interim(df: pd.DataFrame, filename: str = INTERIM_FILE) -> Path:
    """Write merged frame to data/interim/merged/ as parquet."""
    MERGED_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = MERGED_INTERIM_DIR / filename
    df.to_parquet(path, index=False)
    logger.success(f"Wrote {len(df):,} rows to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path
# --------------------------------------------------------------------------
# NOTEBBOK HELPER FUNCTION
# --------------------------------------------------------------------------

def summarise_merge(df: pd.DataFrame) -> pd.Series:
    """Match quality and category spread after merging."""
    return pd.Series({
        "rows": len(df),
        "unmatched_src": df["src_entity_type"].isna().mean(),
        "unmatched_dst": df["dst_entity_type"].isna().mean(),
        "src_entity_types": df["src_entity_type"].nunique(),
        "src_bank_countries": df["src_bank_country"].nunique(),
        "same_country_share": (df["src_bank_country"] == df["dst_bank_country"]).mean(),
        "same_entity_share": (df["src_entity_type"] == df["dst_entity_type"]).mean(),
    })


if __name__ == "__main__":
    build_interim()