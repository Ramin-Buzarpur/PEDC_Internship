# -*- coding: utf-8 -*-

"""
Yahoo Finance Top Crypto Downloader
====================================

Features:
- Fetch Top N cryptocurrencies from Yahoo Finance
- Save metadata as CSV + Parquet
- Fetch Yahoo Finance market table data
- Fetch OHLCV data
- Save OHLCV as CSV + Parquet
- Retry failed requests
- Progress bar
- Run-specific output directory
- Resume support
"""

import os
import time
import json
import random
import requests
import pandas as pd

from datetime import datetime, timezone
from tqdm import tqdm


# =============================================================================
# USER SETTINGS
# =============================================================================

TOP_N = 100

START_DATE = "2026-09-20"
END_DATE = "2026-09-25"

INTERVAL = "15m"

RESUME = True

REQUEST_DELAY = 1.0
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

REQUEST_TIMEOUT = 30


# =============================================================================
# INTERVAL MAP
# =============================================================================

INTERVAL_MAP = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "90m": "90m",
    "1d": "1d",
    "5d": "5d",
    "1wk": "1wk",
    "1mo": "1mo",
    "3mo": "3mo",
}

if INTERVAL not in INTERVAL_MAP:
    raise ValueError(
        f"Invalid interval: {INTERVAL}\n"
        f"Available intervals: {list(INTERVAL_MAP.keys())}"
    )

YAHOO_INTERVAL = INTERVAL_MAP[INTERVAL]


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

BASE_OUTPUT_DIR = r"D:\YahooCrypto"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    f"Run_{RUN_TIMESTAMP}"
)

METADATA_DIR = os.path.join(
    RUN_DIR,
    "metadata"
)

OHLCV_CSV_DIR = os.path.join(
    RUN_DIR,
    "ohlcv_csv"
)

OHLCV_PARQUET_DIR = os.path.join(
    RUN_DIR,
    "ohlcv_parquet"
)

LOG_DIR = os.path.join(
    RUN_DIR,
    "logs"
)


# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(OHLCV_CSV_DIR, exist_ok=True)
os.makedirs(OHLCV_PARQUET_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# =============================================================================
# HEADERS
# =============================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/154.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


# =============================================================================
# DATE TO TIMESTAMP
# =============================================================================

def date_to_timestamp(date_string):
    """
    Convert YYYY-MM-DD to Unix timestamp.
    """

    dt = datetime.strptime(
        date_string,
        "%Y-%m-%d"
    )

    dt = dt.replace(
        tzinfo=timezone.utc
    )

    return int(
        dt.timestamp()
    )


# =============================================================================
# SAFE FILENAME
# =============================================================================

def safe_filename(name):
    """
    Convert symbol to safe Windows filename.
    """

    invalid_chars = '<>:"/\\|?*'

    for char in invalid_chars:
        name = name.replace(char, "_")

    return name


# =============================================================================
# REQUEST WITH RETRY
# =============================================================================

def request_with_retry(
    url,
    params=None
):
    """
    Send HTTP GET request with retry mechanism.
    """

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                return response

            print(
                f"HTTP {response.status_code} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

        except requests.RequestException as e:

            print(
                f"Request error: {e} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

        if attempt < MAX_RETRIES:

            sleep_time = (
                BACKOFF_FACTOR ** (attempt - 1)
            )

            sleep_time += random.uniform(
                0,
                0.5
            )

            time.sleep(
                sleep_time
            )

    return None


# =============================================================================
# FETCH TOP CRYPTO LIST
# =============================================================================

def fetch_top_crypto_list(top_n=500):

    print("\n")
    print("=" * 70)
    print("FETCHING TOP CRYPTO METADATA")
    print("=" * 70)

    url = (
        "https://query1.finance.yahoo.com/"
        "v1/finance/screener/predefined/saved"
    )

    all_quotes = []

    page_size = 250

    for start in range(
        0,
        top_n,
        page_size
    ):

        count = min(
            page_size,
            top_n - start
        )

        params = {
            "scrIds": "all_cryptocurrencies_us",
            "count": count,
            "start": start
        }

        response = request_with_retry(
            url,
            params=params
        )

        if response is None:
            print(
                f"Failed to load "
                f"start={start}"
            )
            continue

        try:

            data = response.json()

            result = (
                data
                .get("finance", {})
                .get("result", [])
            )

            if not result:
                continue

            quotes = (
                result[0]
                .get("quotes", [])
            )

            all_quotes.extend(
                quotes
            )

            print(
                f"Yahoo screener: "
                f"{len(all_quotes)}/{top_n} loaded"
            )

        except Exception as e:

            print(
                f"Metadata parsing error: {e}"
            )

        time.sleep(
            REQUEST_DELAY
        )

    all_quotes = all_quotes[:top_n]

    rows = []

    for rank, quote in enumerate(
        all_quotes,
        start=1
    ):

        rows.append({

            "rank": rank,

            "symbol": quote.get(
                "symbol"
            ),

            "name": (
                quote.get("longName")
                or quote.get("shortName")
            ),

            "price": quote.get(
                "regularMarketPrice"
            ),

            "market_cap": quote.get(
                "marketCap"
            ),

            "volume": quote.get(
                "regularMarketVolume"
            ),

            "currency": quote.get(
                "currency"
            ),

            "exchange": quote.get(
                "exchange"
            ),

            "quote_type": quote.get(
                "quoteType"
            )
        })

    df = pd.DataFrame(
        rows
    )

    return df


# =============================================================================
# FETCH YAHOO FINANCE MARKET TABLE
# =============================================================================

def fetch_yahoo_market_table(top_n=500):

    print("\n")
    print("=" * 80)
    print("FETCHING YAHOO FINANCE CRYPTO MARKET TABLE")
    print("=" * 80)

    url = (
        "https://query1.finance.yahoo.com/"
        "v1/finance/screener/predefined/saved"
    )

    all_rows = []

    # Yahoo Finance page uses 100 items per page
    PAGE_SIZE = 100

    total_pages = (
        top_n + PAGE_SIZE - 1
    ) // PAGE_SIZE

    for page in range(
        total_pages
    ):

        start = page * PAGE_SIZE

        count = min(
            PAGE_SIZE,
            top_n - start
        )

        params = {

            "scrIds":
                "all_cryptocurrencies_us",

            "count":
                count,

            "start":
                start,

            "formatted":
                "true",

            "lang":
                "en-US",

            "region":
                "US",

            "corsDomain":
                "finance.yahoo.com"
        }

        print(
            f"\nFetching page "
            f"{page + 1}/{total_pages} "
            f"(start={start}, count={count})..."
        )

        response = request_with_retry(
            url,
            params=params
        )

        if response is None:

            print(
                f"ERROR: Failed to fetch "
                f"page {page + 1}"
            )

            continue

        try:

            data = response.json()

            finance = data.get(
                "finance",
                {}
            )

            result = finance.get(
                "result",
                []
            )

            if not result:

                print(
                    f"WARNING: Empty result "
                    f"for page {page + 1}"
                )

                continue

            screener = result[0]

            quotes = screener.get(
                "quotes",
                []
            )

            print(
                f"Received "
                f"{len(quotes)} cryptocurrencies."
            )

            for item in quotes:

                # =========================================================
                # HELPER
                # =========================================================

                def get_value(*keys):

                    for key in keys:

                        value = item.get(
                            key
                        )

                        if value is not None:

                            if isinstance(
                                value,
                                dict
                            ):

                                if "raw" in value:
                                    return value[
                                        "raw"
                                    ]

                                if "fmt" in value:
                                    return value[
                                        "fmt"
                                    ]

                            return value

                    return None

                # =========================================================
                # BASIC INFORMATION
                # =========================================================

                symbol = item.get(
                    "symbol"
                )

                name = (
                    item.get("longName")
                    or item.get("shortName")
                )

                # =========================================================
                # PRICE
                # =========================================================

                price = get_value(
                    "regularMarketPrice"
                )

                # =========================================================
                # CHANGE
                # =========================================================

                change = get_value(
                    "regularMarketChange"
                )

                change_percent = get_value(
                    "regularMarketChangePercent"
                )

                # =========================================================
                # MARKET DATA
                # =========================================================

                market_cap = get_value(
                    "marketCap"
                )

                volume = get_value(
                    "regularMarketVolume",
                    "volume"
                )

                volume_in_currency_24h = get_value(
                    "volumeAllCurrencies",
                    "regularMarketVolume"
                )

                total_volume_all_currencies_24h = get_value(
                    "volumeAllCurrencies",
                    "regularMarketVolume"
                )

                circulating_supply = get_value(
                    "circulatingSupply"
                )

                # =========================================================
                # 52 WEEK INFORMATION
                # =========================================================

                fifty_two_week_change = get_value(
                    "fiftyTwoWeekChangePercent"
                )

                fifty_two_week_low = get_value(
                    "fiftyTwoWeekLow"
                )

                fifty_two_week_high = get_value(
                    "fiftyTwoWeekHigh"
                )

                # =========================================================
                # EXCHANGE / CURRENCY
                # =========================================================

                exchange = item.get(
                    "exchange"
                )

                currency = item.get(
                    "currency"
                )

                quote_type = item.get(
                    "quoteType"
                )

                # =========================================================
                # BUILD ROW
                # =========================================================

                row = {

                    "Rank":
                        start + len(all_rows) + 1,

                    "Symbol":
                        symbol,

                    "Name":
                        name,

                    "Price":
                        price,

                    "Change":
                        change,

                    "Change %":
                        change_percent,

                    "Market Cap":
                        market_cap,

                    "Volume":
                        volume,

                    "Volume In Currency (24hr)":
                        volume_in_currency_24h,

                    "Total Volume All Currencies (24hr)":
                        total_volume_all_currencies_24h,

                    "Circulating Supply":
                        circulating_supply,

                    "52 Wk Change %":
                        fifty_two_week_change,

                    "52 Wk Range":
                        (
                            f"{fifty_two_week_low} "
                            f"{fifty_two_week_high}"
                            if (
                                fifty_two_week_low
                                is not None
                                and
                                fifty_two_week_high
                                is not None
                            )
                            else None
                        ),

                    "52 Wk Low":
                        fifty_two_week_low,

                    "52 Wk High":
                        fifty_two_week_high,

                    "Exchange":
                        exchange,

                    "Currency":
                        currency,

                    "Quote Type":
                        quote_type
                }

                all_rows.append(
                    row
                )

        except Exception as e:

            print(
                f"ERROR parsing Yahoo "
                f"market table page "
                f"{page + 1}: {e}"
            )

        if page < total_pages - 1:

            time.sleep(
                REQUEST_DELAY
            )

    # =========================================================================
    # KEEP EXACTLY TOP_N
    # =========================================================================

    all_rows = all_rows[
        :top_n
    ]

    if not all_rows:

        print(
            "\nERROR: No market data received."
        )

        return None

    # =========================================================================
    # REBUILD RANK
    # =========================================================================

    for i, row in enumerate(
        all_rows,
        start=1
    ):

        row["Rank"] = i

    # =========================================================================
    # DATAFRAME
    # =========================================================================

    df = pd.DataFrame(
        all_rows
    )

    # =========================================================================
    # NUMERIC COLUMNS
    # =========================================================================

    numeric_columns = [

        "Rank",

        "Price",

        "Change",

        "Change %",

        "Market Cap",

        "Volume",

        "Volume In Currency (24hr)",

        "Total Volume All Currencies (24hr)",

        "Circulating Supply",

        "52 Wk Change %",

        "52 Wk Low",

        "52 Wk High"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    csv_path = os.path.join(
        METADATA_DIR,
        f"yahoo_market_data_{top_n}.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =========================================================================
    # SAVE PARQUET
    # =========================================================================

    parquet_path = os.path.join(
        METADATA_DIR,
        f"yahoo_market_data_{top_n}.parquet"
    )

    df.to_parquet(
        parquet_path,
        index=False
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("YAHOO MARKET TABLE COMPLETED")
    print("=" * 80)

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print("\nCSV:")
    print(
        csv_path
    )

    print("\nParquet:")
    print(
        parquet_path
    )

    print("\nColumns:")

    for i, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i:02d}. {column}"
        )

    print("\nFirst 5 rows:")

    print(
        df[
            [
                "Rank",
                "Symbol",
                "Name",
                "Price",
                "Change",
                "Change %",
                "Market Cap"
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )

    return df


# =============================================================================
# FETCH OHLCV
# =============================================================================

def fetch_ohlcv(
    symbol,
    start_date,
    end_date,
    interval
):

    url = (
        "https://query1.finance.yahoo.com/"
        f"v8/finance/chart/{symbol}"
    )

    # =========================================================================
    # START TIMESTAMP
    # =========================================================================

    period1 = date_to_timestamp(
        start_date
    )

    # =========================================================================
    # END TIMESTAMP
    #
    # اگر END_DATE برابر تاریخ امروز باشد،
    # زمان فعلی UTC به عنوان انتهای بازه استفاده می‌شود.
    #
    # بنابراین:
    #
    # START_DATE = "2026-09-30"
    # END_DATE   = "2026-09-30"
    #
    # به معنی:
    #
    # 2026-09-30 00:00:00
    # تا همین لحظه
    #
    # =========================================================================

    today_utc = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d"
    )

    if end_date == today_utc:

        period2 = int(
            datetime.now(
                timezone.utc
            ).timestamp()
        )

    else:

        period2 = date_to_timestamp(
            end_date
        )

        # END_DATE شامل کل آن روز باشد
        period2 += 24 * 60 * 60

    # =========================================================================
    # PARAMS
    # =========================================================================

    params = {

        "period1":
            period1,

        "period2":
            period2,

        "interval":
            interval,

        "events":
            "history",

        "includeAdjustedClose":
            "true"
    }

    response = request_with_retry(
        url,
        params=params
    )

    if response is None:
        return None

    try:

        data = response.json()

        chart = data.get(
            "chart",
            {}
        )

        result = chart.get(
            "result"
        )

        if not result:
            return None

        result = result[0]

        timestamps = result.get(
            "timestamp",
            []
        )

        indicators = result.get(
            "indicators",
            {}
        )

        quote = indicators.get(
            "quote",
            [{}]
        )[0]

        adjclose_data = indicators.get(
            "adjclose",
            [{}]
        )

        if adjclose_data:

            adjclose = adjclose_data[0].get(
                "adjclose",
                []
            )

        else:

            adjclose = []

        rows = []

        for i, timestamp in enumerate(
            timestamps
        ):

            dt = datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc
            )

            rows.append({

                "Datetime":
                    dt,

                "Open":
                    (
                        quote.get(
                            "open",
                            [None]
                        )[i]

                        if i < len(
                            quote.get(
                                "open",
                                []
                            )
                        )

                        else None
                    ),

                "High":
                    (
                        quote.get(
                            "high",
                            [None]
                        )[i]

                        if i < len(
                            quote.get(
                                "high",
                                []
                            )
                        )

                        else None
                    ),

                "Low":
                    (
                        quote.get(
                            "low",
                            [None]
                        )[i]

                        if i < len(
                            quote.get(
                                "low",
                                []
                            )
                        )

                        else None
                    ),

                "Close":
                    (
                        quote.get(
                            "close",
                            [None]
                        )[i]

                        if i < len(
                            quote.get(
                                "close",
                                []
                            )
                        )

                        else None
                    ),

                "Adj Close":
                    (
                        adjclose[i]

                        if i < len(
                            adjclose
                        )

                        else None
                    ),

                "Volume":
                    (
                        quote.get(
                            "volume",
                            [None]
                        )[i]

                        if i < len(
                            quote.get(
                                "volume",
                                []
                            )
                        )

                        else None
                    )
            })

        df = pd.DataFrame(
            rows
        )

        return df

    except Exception as e:

        print(
            f"OHLCV parsing error "
            f"for {symbol}: {e}"
        )

        return None


# =============================================================================
# SAVE OHLCV
# =============================================================================

def save_ohlcv(
    df,
    symbol
):

    filename = safe_filename(
        symbol
    )

    csv_path = os.path.join(
        OHLCV_CSV_DIR,
        f"{filename}.csv"
    )

    parquet_path = os.path.join(
        OHLCV_PARQUET_DIR,
        f"{filename}.parquet"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    df.to_parquet(
        parquet_path,
        index=False
    )

    return (
        csv_path,
        parquet_path
    )


# =============================================================================
# EXISTING DATA AVAILABLE
# =============================================================================

def existing_data_available(
    symbol
):

    filename = safe_filename(
        symbol
    )

    csv_path = os.path.join(
        OHLCV_CSV_DIR,
        f"{filename}.csv"
    )

    parquet_path = os.path.join(
        OHLCV_PARQUET_DIR,
        f"{filename}.parquet"
    )

    return (
        os.path.exists(csv_path)
        or
        os.path.exists(parquet_path)
    )


# =============================================================================
# SAVE RUN CONFIG
# =============================================================================

def save_run_config():

    config = {

        "top_n":
            TOP_N,

        "start_date":
            START_DATE,

        "end_date":
            END_DATE,

        "interval":
            INTERVAL,

        "yahoo_interval":
            YAHOO_INTERVAL,

        "resume":
            RESUME,

        "request_delay":
            REQUEST_DELAY,

        "max_retries":
            MAX_RETRIES,

        "backoff_factor":
            BACKOFF_FACTOR,

        "run_timestamp":
            RUN_TIMESTAMP,

        "run_directory":
            RUN_DIR
    }

    config_path = os.path.join(
        RUN_DIR,
        "config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nConfig saved:")
    print(
        config_path
    )


# =============================================================================
# DOWNLOAD ALL CRYPTO DATA
# =============================================================================

def download_all_crypto_data(
    metadata_df
):

    print("\n")
    print("=" * 70)
    print("DOWNLOADING OHLCV DATA")
    print("=" * 70)

    results = []

    for _, row in tqdm(
        metadata_df.iterrows(),
        total=len(metadata_df),
        desc="Downloading"
    ):

        symbol = row[
            "symbol"
        ]

        if not symbol:
            continue

        # ================================================================
        # RESUME
        # ================================================================

        if RESUME:

            if existing_data_available(
                symbol
            ):

                results.append({

                    "symbol":
                        symbol,

                    "status":
                        "skipped_existing"
                })

                continue

        # ================================================================
        # DOWNLOAD
        # ================================================================

        df = fetch_ohlcv(
            symbol,
            START_DATE,
            END_DATE,
            YAHOO_INTERVAL
        )

        if df is None or df.empty:

            results.append({

                "symbol":
                    symbol,

                "status":
                    "failed",

                "rows":
                    0
            })

            continue

        # ================================================================
        # SAVE
        # ================================================================

        csv_path, parquet_path = save_ohlcv(
            df,
            symbol
        )

        results.append({

            "symbol":
                symbol,

            "status":
                "success",

            "rows":
                len(df),

            "csv":
                csv_path,

            "parquet":
                parquet_path
        })

        time.sleep(
            REQUEST_DELAY
        )

    return pd.DataFrame(
        results
    )


# =============================================================================
# SAVE DOWNLOAD LOGS
# =============================================================================

def save_download_logs(
    logs_df
):

    csv_path = os.path.join(
        LOG_DIR,
        "download_log.csv"
    )

    parquet_path = os.path.join(
        LOG_DIR,
        "download_log.parquet"
    )

    logs_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    logs_df.to_parquet(
        parquet_path,
        index=False
    )

    print("\nLogs saved:")
    print(
        csv_path
    )
    print(
        parquet_path
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")
    print("=" * 70)
    print("YAHOO FINANCE CRYPTO DOWNLOADER")
    print("=" * 70)

    print(
        f"Run directory: {RUN_DIR}"
    )

    print(
        f"Top N        : {TOP_N}"
    )

    print(
        f"Start date   : {START_DATE}"
    )

    print(
        f"End date     : {END_DATE}"
    )

    print(
        f"Interval     : {INTERVAL}"
    )

    print(
        f"Yahoo        : {YAHOO_INTERVAL}"
    )

    print("=" * 70)

    # =====================================================================
    # CONFIG
    # =====================================================================

    save_run_config()

    # =====================================================================
    # TOP CRYPTO METADATA
    # =====================================================================

    metadata_df = fetch_top_crypto_list(
        TOP_N
    )

    if metadata_df is None or metadata_df.empty:

        print(
            "\nERROR: Could not retrieve "
            "crypto metadata."
        )

        return

    # =====================================================================
    # SAVE ORIGINAL METADATA
    # =====================================================================

    metadata_csv = os.path.join(
        METADATA_DIR,
        f"top_{TOP_N}_crypto.csv"
    )

    metadata_parquet = os.path.join(
        METADATA_DIR,
        f"top_{TOP_N}_crypto.parquet"
    )

    metadata_df.to_csv(
        metadata_csv,
        index=False,
        encoding="utf-8-sig"
    )

    metadata_df.to_parquet(
        metadata_parquet,
        index=False
    )

    print("\nMetadata saved:")
    print(
        metadata_csv
    )
    print(
        metadata_parquet
    )

    # =====================================================================
    # YAHOO MARKET TABLE
    # =====================================================================

    yahoo_market_df = fetch_yahoo_market_table(
        TOP_N
    )

    # =====================================================================
    # OHLCV
    # =====================================================================

    logs_df = download_all_crypto_data(
        metadata_df
    )

    # =====================================================================
    # LOGS
    # =====================================================================

    if logs_df is not None and not logs_df.empty:

        save_download_logs(
            logs_df
        )

    # =====================================================================
    # FINISH
    # =====================================================================

    print("\n")
    print("=" * 70)
    print("DOWNLOAD COMPLETED")
    print("=" * 70)

    print(
        f"Run directory:"
    )

    print(
        RUN_DIR
    )

    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()