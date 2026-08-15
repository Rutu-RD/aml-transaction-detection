'''loading ,cleaning and normalising raw IBM Aml transaction data.'''
from __future__ import annotations
 
from pathlib import Path
 
import pandas as pd
from loguru import logger
 
from aml_detection.config import RAW_DATA_DIR,INTERIM_DATA_DIR
COLUMN_MAP: dict[str, str] = {
    "Timestamp": "timestamp",
    "From Bank": "src_bank",
    "Account": "src_account",
    "To Bank": "dst_bank",
    "Account.1": "dst_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "currency_received",
    "Amount Paid": "amount_paid",
    "Payment Currency": "currency_paid",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "currency_received",
    "currency_paid",
    "payment_format",
)
class DatasetError(Exception):
    """Base class for dataset problems."""
 
 
class DatasetNotFoundError(DatasetError):
    """Raised when the expected raw file is not on disk."""
 
 
class SchemaValidationError(DatasetError):
    """Raised when the file loaded but does not have the expected columns."""

def _resolve_path(filename: str, data_dir: Path | None = None) -> Path:
    directory = data_dir or RAW_DATA_DIR
    return directory / filename


def _validate_schema(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Fail loudly and usefully if the file is not the shape we expect."""
    missing = [column for column in COLUMN_MAP if column not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"{source.name} is missing expected columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )
    return df

 
def load_raw(filename: str = "HI-Small_Trans.csv",nrows: int | None = None,data_dir: Path | None = None,) -> pd.DataFrame:
    '''
    reads the raw transaction csv from disk unmodified apart from validating schema
    parameters
    filename: name of csv inside raw directory
    nrows:caps number of rows to be read .only for development real file is 5M rows
    data_dir: override the directory mainly for testing
    
    raises errors
    DatasetNotFoundError:file is not on disk
    SchemaValidationError:file exists but columns are unexpected
    '''
    path = _resolve_path(filename, data_dir)
    logger.info(f"Reading {path}" + (f" (first {nrows:,} rows)" if nrows else ""))
 
    try:
        df = pd.read_csv(path, nrows=nrows)
    except FileNotFoundError as exc:
        raise DatasetNotFoundError(
            f"Raw data not found at {path}.\n"
            "Download HI-Small_Trans.csv from Kaggle and place it in "
            f"{path.parent}/"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"{path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetError(f"{path} could not be parsed as CSV: {exc}") from exc
 
    logger.success(f"Read {len(df):,} rows x {df.shape[1]} columns")
    return _validate_schema(df, path)


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_MAP)
 
 
def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
 
 
def _namespace_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix account ids with their bank.
 
    The same account id can exist at more than one bank. Without this, two
    unrelated accounts merge into one node and we invent transaction links
    that never happened.
    """
    return df.assign(
        src_account=lambda d: d["src_bank"].astype(str) + "_" + d["src_account"].astype(str),
        dst_account=lambda d: d["dst_bank"].astype(str) + "_" + d["dst_account"].astype(str),
    )


def _flag_self_loops(df: pd.DataFrame) -> pd.DataFrame:
    """Mark transactions where sender and receiver are the same account.
 
    These are frequently 'Reinvestment' rows. They are not transfers between
    parties, so they distort account-level aggregates (counterparty counts,
    outgoing volume) if treated like ordinary payments. Flagged rather than
    dropped so the decision stays explicit and reversible.
    """
    return df.assign(is_self_loop=lambda d: d["src_account"] == d["dst_account"])
 
 
def _add_amount_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Difference between paid and received, which is non-zero on FX legs."""
    return df.assign(
        is_cross_currency=lambda d: d["currency_paid"] != d["currency_received"],
        amount_delta=lambda d: d["amount_paid"] - d["amount_received"],
    )
 
 
def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink memory footprint. Matters at 5M rows."""
    return df.astype(
        {column: "category" for column in CATEGORICAL_COLUMNS if column in df.columns}
        | {"is_laundering": "int8"}
    )
 
 
def _sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Order by time so temporal splits and rolling features are well defined."""
    return df.sort_values("timestamp").reset_index(drop=True)


# The raw file spans 2022-09-01 to 2022-09-18, but the two regions are not
# comparable:
#
#   days 1-10   ~500,000 txns/day, laundering rate ~0.1%
#   days 11-18  ~400 down to 11 txns/day, laundering rate ~58-73%
#
# Background transaction generation stops after day 10; what remains is the
# trailing legs of laundering chains that started earlier. Those 1,108 rows
# (0.02% of the file) are excluded because:
#
#   1. A temporal split would put them in the test set, where a 58% positive
#      rate produces metrics that look excellent and mean nothing.
#   2. Account features would learn "active after 11 Sept" == laundering,
#      this is not correct
# The chains are not lost: their earlier legs remain in days 1-10.

SIMULATION_END = pd.Timestamp("2022-09-11")

# Accounts have no prior history on day 1, so "activity in the last N days"
# features are systematically zero for early rows. These days are kept for
# computing history but flagged so they can be dropped from training rows.
BURN_IN_DAYS = 2

def _drop_simulation_tail(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows after background generation stops (see SIMULATION_END)."""
    before = len(df)
    trimmed = df[df["timestamp"] < SIMULATION_END]
    dropped = before - len(trimmed)
    if dropped:
        logger.info(
            f"Dropped {dropped:,} tail rows on/after "
            f"{SIMULATION_END.date()} ({dropped / before:.3%} of file)"
        )
    return trimmed


def _flag_burn_in(df: pd.DataFrame) -> pd.DataFrame:
    """Mark early rows whose account-history features cannot be trusted."""
    burn_in_end = df["timestamp"].min().normalize() + pd.Timedelta(days=BURN_IN_DAYS)
    return df.assign(is_burn_in=lambda d: d["timestamp"] < burn_in_end)



def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning step, in order, to a raw frame.
 
    Composed with .pipe so the sequence reads top to bottom and any step can be
    commented out or reordered without touching the others.
 
    
    """
    logger.info("Cleaning transactions")
 
    cleaned = (
        df.pipe(_rename_columns)
        .pipe(_parse_timestamp)
        .pipe(_drop_simulation_tail)
        .pipe(_namespace_accounts)
        .pipe(_flag_self_loops)
        .pipe(_add_amount_delta)
        .pipe(_downcast)
        .pipe(_sort_chronologically)
        .pipe(_flag_burn_in)
    )
 
    logger.success( 
        f"Cleaned {len(cleaned):,} rows | "
        f"laundering {cleaned['is_laundering'].mean():.4%} | "
        f"self-loops {cleaned['is_self_loop'].mean():.1%}"
    )
    return cleaned

def load_dataset(
    filename: str = "HI-Small_Trans.csv",
    nrows: int | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load and clean in one call."""
    return (
        clean(load_raw(filename=filename, nrows=nrows, data_dir=data_dir))    
    )

def summarise(df: pd.DataFrame) -> pd.Series:
    """Structural facts worth knowing before making any modelling decision."""
    accounts = pd.unique(
        pd.concat([df["src_account"], df["dst_account"]], ignore_index=True)
    )
   
    return pd.Series(
        {
            "rows": len(df),
            "unique_accounts": len(accounts),
            "laundering_rate": df["is_laundering"].mean(),
            "laundering_count": int(df["is_laundering"].sum()),
            "self_loop_share": df["is_self_loop"].mean(),
            "cross_currency_share": df["is_cross_currency"].mean(),
            "start": df["timestamp"].min(),
            "end": df["timestamp"].max(),
            "days_covered": (df["timestamp"].max() - df["timestamp"].min()).days,
        }
    )

# --------------------------------------------------------------------------
# saving the loaded and cleaned data to interim/transaction/filename

# --------------------------------------------------------------------------

TRANSACTION_INTERIM_DIR = INTERIM_DATA_DIR / "transaction"


def save_data_transaction(df: pd.DataFrame, filename: str) -> Path:
    """Write the cleaned transaction frame to data/interim/transaction/."""
    TRANSACTION_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    save_path = TRANSACTION_INTERIM_DIR / filename
    df.to_parquet(save_path, index=False)
    logger.success(
        f"Wrote {len(df):,} rows to {save_path} "
        f"({save_path.stat().st_size / 1e6:.1f} MB)"
    )
    return save_path
    
if __name__=="__main__":
    transactions_df=load_dataset("HI-Small_Trans.csv")
    
    save_data_transaction(transactions_df,"transaction_cleaned.parquet")
    logger.info("saved cleaned transaction data to interim succesfully")


