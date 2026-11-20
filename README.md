# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lucaslml0/gta6-trophy-forecast-20261120/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lucaslml0/gta6-trophy-forecast-20261120/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at a case's target time — entered in make.fun's
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty ($250 in USDC).

## The program

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
one `<id>.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).
Each output is `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`
— five quantiles of trophy `n`'s earn rate at the target, in percent.

## The model

Exophase refreshes a game's counts on its own schedule, so a case's history
is long runs of identical reads punctuated by jumps. The forecast is built
from what that data actually supports:

1. **Collapse identical reads.** Consecutive reads with the same
   players-tracked count and rates are one point, not hundreds — otherwise
   the flat stretches would swamp any fit.
2. **Trend per trophy, shrunk.** Fit `logit(rate)` against
   `ln(1 + hours since release)` by weighted least squares over the distinct
   reads (recent reads count more), then shrink the slope toward its tier
   mean by its t-statistic. A flat or noisy history falls back to
   persistence; a clear trend is followed.
3. **Honest spread.** The 5–95% band combines a random-walk term that grows
   with the horizon, the trend's standard error times the extrapolation
   distance, and the size of the refresh jumps the history actually shows.
   A trophy every tracked player holds (≈100%) stays ≈100%.
4. **Guardrails.** The median never moves more than 3 logit units from the
   last read, quantiles are rounded to 4 decimals, non-decreasing, and
   clamped to [0.01, 100].

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=../gta6-trophy-rarity-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs write the same bytes. On the kit's practice cases (`p00`, `p01`)
the history is a single distinct read, so the forecast is persistence with
a refresh-aware spread — the practice pays nothing; the final case carries
the real 2.5-day trajectory the trend fit is built for.

## Files

- `main.py` — the forecaster (the entrypoint the grader runs).
- `.makefun/owner.json` — proves the repository is the entrant's.
- `LICENSE` — MIT.
