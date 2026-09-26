# Acquiring the data (ThetaData)

The store is built from [ThetaData](https://www.thetadata.net/) historical data: a paid,
licensed feed. **No market data is included in this repository**; everything below needs
your own subscription. To see the pipeline work without one, run `spx-pipeline demo`.

## 1. Run the ThetaData Terminal

The Terminal is a local Java process that authenticates with your account and serves the
API on `127.0.0.1:25503`. `launch_terminal.sh` starts it in the background and waits until
it answers; it reads your credentials from a `creds.txt` file next to the jar (email on
line 1, password on line 2). **Never commit that file.**

```bash
spx-pipeline thetadata ping          # is the Terminal up?
```

## 2. Download

```bash
pip install -e ".[thetadata]"
export THETADATA_ROOT=~/thetadata    # where monthly files are written
spx-pipeline thetadata download                   # everything, 2019 → today
spx-pipeline thetadata download --dataset greeks  # one family
spx-pipeline thetadata download --check           # verify the files already on disk
```

The downloader runs requests concurrently under a semaphore (HTTP slots are released
during back-off sleeps), retries with exponential back-off, waits out rate limits, and
journals every completed month so an interrupted run resumes where it stopped. One
`requests.Session` per thread; files are written as ZSTD Parquet with an explicit schema,
and empty months still get an empty file with the right schema so a resume can tell
"done, nothing there" from "not done".

## 3. Validate the raw files

```bash
spx-pipeline thetadata validate-raw --quick
```

About thirty checks, each PASS / WARN / FAIL: schemas, business-day coverage, the SPXW
expiration calendar (Monday/Wednesday/Friday only before May 2022, daily after), market
hours, bid ≤ ask, strikes within a plausible band of the underlying, delta in [-1, 1],
spread outliers, duplicates, index ↔ options consistency of the underlying price, and the
feed's known quirks — OPRA test contracts expiring in 1882, zero prices at 09:30.

## 4. Build the store and check it

```bash
spx-pipeline thetadata split --store-root ~/thetadata   # monthly vendor files → daily store
spx-pipeline thetadata validate-store --verbose
```

The splitter partitions each monthly file by session, casts to the registry schema, drops
exchange/condition codes and writes one ZSTD file per session under
`store/<dataset>/year=YYYY/month=MM/`. From then on, `DataLoader(store_root=~/thetadata)`
reads it.
