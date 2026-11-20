"""Forecast each GTA VI PS5 trophy's Exophase earn rate at a case's target.

Model (written from the case data, standard library only, deterministic):
  * Exophase refreshes a game's counts on its own schedule, so the history is
    long runs of identical reads punctuated by jumps. Only *distinct* reads
    carry information: consecutive reads with the same players-tracked count
    and the same rates are collapsed to one point before any fit.
  * For each trophy, fit logit(rate) against x = ln(1 + hours since release)
    by weighted least squares over the distinct reads (weights favour recent
    reads), then shrink the slope toward the tier mean by its t-statistic:
    a flat or noisy history falls back to persistence, a clear trend is
    followed. The median at the target is the shrunken trend read at the
    target's x, blended with the last read's rate when the trend is weak.
  * The spread is honest about what can move a rate: a random-walk term that
    grows with the horizon, the trend's own standard error times the
    extrapolation distance, and the size of the refresh jumps the history
    actually shows. A trophy at 100% in every read stays ~100%.
  * Everything is clamped to sane bounds (a median never moves more than
    3 logit units from the last read) so a wild extrapolation cannot sink
    the entry.

Run: `python3 main.py` reads every *.in in $MAKEFUN_INPUT_DIR (else /input)
and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output). No network, no
randomness: two runs write the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Random-walk volatility of a rate's logit, per sqrt-day (house baseline: 0.5).
SIGMA_RW = 0.5
# A median never moves this far (logit units) from the last read.
MAX_MOVE = 3.0
# Spread floor / cap for a refresh jump (logit units).
JUMP_FLOOR = 0.15
JUMP_CAP = 1.0


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (as the house baseline rounds it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def distinct_reads(history):
    """Collapse runs of identical reads to one point each (plus first/last).

    Exophase holds a page unchanged between refreshes, so a history can hold
    hundreds of identical reads. They carry no information beyond the first
    and would swamp any fit if counted one by one.
    """
    out = []
    for h in history:
        key = (h.get("players_tracked"), tuple(h["rates"]))
        if out and out[-1][0] == key:
            # Same page as the previous kept read: keep the later timestamp
            # only if it is the newest read overall (handled by caller).
            continue
        out.append((key, h))
    # Always keep the very last read as its own point (it is the freshest).
    if history:
        last = history[-1]
        key = (last.get("players_tracked"), tuple(last["rates"]))
        if not out or out[-1][0] != key:
            out.append((key, last))
    return [h for _, h in out]


def wls(xs, ys, ws):
    """Weighted least squares for y = a + b*x. Returns (a, b, se_b, n)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, len(xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    # Residual variance and the slope's standard error.
    n = len(xs)
    if n > 2:
        rss = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        dof = max(1, n - 2)
        s2 = rss / dof / (sw / n)  # unweight the scale back to per-point
        se_b = math.sqrt(max(s2, 0.0) / sxx)
    else:
        se_b = 0.0
    return (a, b, se_b, n)


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    # Time origin: the game's release (00:00 UTC that day), or the first read.
    release = case.get("game", {}).get("release_date")
    try:
        origin = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        origin = parse_time(history[0]["at"])
    target_t = parse_time(case["target"])
    x_target = math.log1p(max(0.0, (target_t - origin).total_seconds() / 3600.0))

    reads = distinct_reads(history)
    # Per-read x and weight (recent reads count more).
    read_x = []
    for h in reads:
        hrs = max(0.0, (parse_time(h["at"]) - origin).total_seconds() / 3600.0)
        read_x.append(math.log1p(hrs))
    last_read = history[-1]
    last_rates = last_read["rates"]

    # Refresh jumps: median |Δlogit| between consecutive distinct reads,
    # over all trophies. This is how far one Exophase refresh can move a rate.
    jumps = []
    for j in range(1, len(reads)):
        prev, cur = reads[j - 1]["rates"], reads[j]["rates"]
        for a, b in zip(prev, cur):
            if a is not None and b is not None:
                jumps.append(abs(logit(b) - logit(a)))
    jumps.sort()
    jump = jumps[len(jumps) // 2] if jumps else 0.0
    jump = min(JUMP_CAP, max(JUMP_FLOOR, jump))

    # Fit every trophy's trend, then shrink each slope toward its tier mean.
    fits = []
    tier_slopes = {}
    for i, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        for k, h in enumerate(reads):
            r = h["rates"][i]
            if r is None:
                continue
            xs.append(read_x[k])
            ys.append(logit(r))
            # Recency weight: newest read 1.0, older ones decay by half
            # per ~3.2 units of log-time (about the full span of a case's
            # history), so the fit leans recent without discarding the past.
            age_x = max(0.0, read_x[-1] - read_x[k])
            ws.append(0.5 ** (age_x / math.log(25.0)))
        fit = wls(xs, ys, ws) if len(xs) >= 2 else None
        fits.append(fit)
        if fit is not None:
            tier_slopes.setdefault(trophy["tier"], []).append(fit[1])
    tier_mean = {t: sum(v) / len(v) for t, v in tier_slopes.items()}
    global_mean = (
        sum(sum(v) for v in tier_slopes.values()) / sum(len(v) for v in tier_slopes.values())
        if tier_slopes
        else 0.0
    )

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i]
        # A trophy every tracked player holds stays (nearly) universal.
        if last_rate is not None and last_rate >= 100.0 and all(
            h["rates"][i] is not None and h["rates"][i] >= 100.0 for h in reads
        ):
            rows.append({"id": trophy["id"], "q": [99.5, 99.9, 100, 100, 100]})
            continue
        base = last_rate if last_rate is not None else 0.0
        lg_last = logit(base)
        fit = fits[i]
        if fit is None:
            median_lg, se_term = lg_last, 0.0
        else:
            a, b, se_b, n = fit
            t_stat = abs(b) / se_b if se_b > 1e-12 else (99.0 if b else 0.0)
            # Shrink toward the tier mean by significance; a flat history
            # (n < 3 distinct reads) gives no trend at all.
            if n < 3:
                shrink = 0.0
            else:
                shrink = (t_stat * t_stat) / (t_stat * t_stat + 1.0)
            prior = tier_mean.get(trophy["tier"], global_mean)
            b_use = shrink * b + (1.0 - shrink) * prior
            trend_lg = a + b_use * x_target
            # Blend the trend with persistence by how much we trust it.
            w = shrink if n >= 3 else 0.0
            median_lg = w * trend_lg + (1.0 - w) * lg_last
            se_term = se_b * abs(x_target - read_x[-1]) if n >= 3 else 0.0
        # Never let the median run away from the last read.
        median_lg = min(lg_last + MAX_MOVE, max(lg_last - MAX_MOVE, median_lg))
        spread = math.sqrt(
            (SIGMA_RW * math.sqrt(horizon_days)) ** 2 + se_term ** 2 + jump ** 2
        )
        spread = max(0.12, spread)
        qs = []
        for z in Z:
            v = logistic(median_lg + z * spread)
            qs.append(round4(min(100.0, max(0.01, v))))
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
