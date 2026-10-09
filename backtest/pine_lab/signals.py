"""
Python ports of the entry rules from the Arun K Bhaskar Pine Script collection
(github.com/ArunKBhaskar/PineScript, MPL-2.0), default settings, daily bars.

Every function takes numpy arrays o, h, l, c, v (oldest -> newest) and returns
{variant_name: (long_signal, short_signal)} boolean arrays. A signal on bar i
is known at the close of bar i (no look-ahead); the engine enters at bar i+1 open.
"""
import numpy as np
import pandas as pd

NAN = np.nan


# ───────────────────────── indicator helpers (TradingView semantics) ─────────────────────────

def sma(x, n):
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def rma(x, n):
    """Wilder's moving average, seeded with an SMA like ta.rma."""
    x = np.asarray(x, float)
    out = np.full(len(x), NAN)
    alpha = 1.0 / n
    start = None
    for i in range(len(x)):
        if start is None:
            if i >= n - 1 and not np.isnan(x[i - n + 1:i + 1]).any():
                out[i] = x[i - n + 1:i + 1].mean()
                start = i
        else:
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def true_range(h, l, c):
    c1 = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
    tr[0] = h[0] - l[0]
    return tr


def atr(h, l, c, n=14):
    return rma(true_range(h, l, c), n)


def rsi(c, n=14):
    d = np.diff(c, prepend=NAN)
    up = rma(np.where(d > 0, d, 0.0)[1:], n)
    dn = rma(np.where(d < 0, -d, 0.0)[1:], n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / dn)))
    return np.concatenate([[NAN], r])


def mfi(h, l, c, v, n=14):
    src = (h + l + c) / 3
    ch = np.diff(src, prepend=NAN)
    upper = pd.Series(np.where(ch > 0, v * src, 0.0)).rolling(n).sum().to_numpy()
    lower = pd.Series(np.where(ch < 0, v * src, 0.0)).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lower == 0, 100.0, 100 - 100 / (1 + upper / lower))


def cci(h, l, c, n=20):
    src = pd.Series((h + l + c) / 3)
    m = src.rolling(n).mean()
    dev = src.rolling(n).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((src - m) / (0.015 * dev)).to_numpy()


def supertrend(h, l, c, factor=3.0, period=10):
    """Returns (supertrend line, direction) with ta.supertrend semantics (-1 = up)."""
    a = atr(h, l, c, period)
    hl2 = (h + l) / 2
    n = len(c)
    up = np.full(n, NAN); dn = np.full(n, NAN)
    st = np.full(n, NAN); dr = np.full(n, 1)
    for i in range(n):
        if np.isnan(a[i]):
            continue
        u = hl2[i] - factor * a[i]
        d = hl2[i] + factor * a[i]
        if i > 0 and not np.isnan(up[i - 1]):
            u = max(u, up[i - 1]) if c[i - 1] > up[i - 1] else u
            d = min(d, dn[i - 1]) if c[i - 1] < dn[i - 1] else d
        up[i], dn[i] = u, d
        if i == 0 or np.isnan(a[i - 1]):
            dr[i] = 1
        elif st[i - 1] == dn[i - 1]:
            dr[i] = -1 if c[i] > d else 1
        else:
            dr[i] = 1 if c[i] < u else -1
        st[i] = u if dr[i] == -1 else d
    return st, dr


def cross_over(a, b):
    a1, b1 = np.roll(a, 1), np.roll(b, 1)
    r = (a > b) & (a1 <= b1)
    r[0] = False
    return r & ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(a1) & ~np.isnan(b1)


def cross_under(a, b):
    return cross_over(-np.asarray(a, float), -np.asarray(b, float))


def _xo(a_prev, b_prev, a, b):
    """Scalar crossover used inside loops (na -> False)."""
    if a_prev is None or b_prev is None or a is None or b is None:
        return False
    if any(np.isnan(x) for x in (a_prev, b_prev, a, b)):
        return False
    return a > b and a_prev <= b_prev


# ───────────────────────── 1. ICT Displacement Candles ─────────────────────────

def displacement(o, h, l, c, v, min_bars=3, dev=0.60, fib=0.5):
    n = len(c)
    rng = h - l
    thr = dev * sma(rng, 14)
    c1 = np.roll(c, 1); c1[0] = NAN
    bull = (rng >= thr) & (c > o) & (c > c1)
    bear = (rng >= thr) & (c < o) & (c < c1)

    disp_L = np.zeros(n, bool); disp_S = np.zeros(n, bool)
    ret_L = np.zeros(n, bool); ret_S = np.zeros(n, bool)

    bullCnt = bearCnt = 0
    bH, bL, sH, sL = [], [], [], []
    bullFib = bearFib = NAN
    prev_bullFib = prev_bearFib = NAN
    bullActive = bearActive = bullDone = bearDone = False
    prev_bullCnt = prev_bearCnt = 0

    for i in range(n):
        if np.isnan(thr[i]):
            prev_bullFib, prev_bearFib = bullFib, bearFib
            continue
        if bull[i]:
            bullCnt += 1; bearCnt = 0
            bH.append(h[i]); bL.append(l[i])
            if bullCnt == 1:
                sH.clear(); sL.clear()
        elif bear[i]:
            bearCnt += 1; bullCnt = 0
            sH.append(h[i]); sL.append(l[i])
            if bearCnt == 1:
                bH.clear(); bL.clear()
        else:
            bullCnt = bearCnt = 0
            bH.clear(); bL.clear(); sH.clear(); sL.clear()

        if bullCnt >= min_bars:
            hi, lo = max(bH), min(bL)
            bullFib = hi - (hi - lo) * fib
        if bearCnt >= min_bars:
            hi, lo = max(sH), min(sL)
            bearFib = lo + (hi - lo) * fib

        disp_S[i] = prev_bearCnt >= min_bars and bearCnt == 0
        disp_L[i] = prev_bullCnt >= min_bars and bullCnt == 0

        if bullCnt >= min_bars and prev_bullCnt < min_bars:
            bullActive, bearActive, bullDone, bearDone = True, False, False, False
        if bearCnt >= min_bars and prev_bearCnt < min_bars:
            bearActive, bullActive, bearDone, bullDone = True, False, False, False

        if i > 0:
            br = (bearActive and not bearDone and _xo(h[i - 1], prev_bearFib, h[i], bearFib)
                  and h[i - 1] < bearFib)
            bu = (bullActive and not bullDone and _xo(-l[i - 1], -prev_bullFib, -l[i], -bullFib)
                  and l[i - 1] > bullFib)
            if br:
                bearDone = True; ret_S[i] = True
            if bu:
                bullDone = True; ret_L[i] = True

        prev_bullCnt, prev_bearCnt = bullCnt, bearCnt
        prev_bullFib, prev_bearFib = bullFib, bearFib

    return {"Displacement": (disp_L, disp_S), "Displacement Retracement": (ret_L, ret_S)}


# ───────────────────────── 2. ICT Fair Value Gap — First Touch Confirmed ─────────────────────────

def fvg(o, h, l, c, v, dev=0.70):
    n = len(c)
    thr = dev * sma(h - l, 14)
    h2 = np.roll(h, 2); l2 = np.roll(l, 2)
    up = (l > h2) & (l - h2 >= thr); up[:2] = False
    dn = (h < l2) & (l2 - h >= thr); dn[:2] = False
    up &= ~np.isnan(thr); dn &= ~np.isnan(thr)

    upTop = np.full(n, NAN); dnBot = np.full(n, NAN)
    t = b = NAN
    for i in range(n):
        if up[i]: t = l[i]
        if dn[i]: b = h[i]
        upTop[i] = t; dnBot[i] = b

    L2 = cross_under(l, upTop)      # first touch of the bullish gap
    S2 = cross_over(h, dnBot)
    sigL = np.zeros(n, bool); sigS = np.zeros(n, bool)
    stL = stS = 0
    l2_high = s2_low = NAN
    for i in range(1, n):
        # long
        if L2[i]:
            l2_prev, l2_high = l2_high, h[i]
        else:
            l2_prev = l2_high
        conf = (not np.isnan(l2_high) and c[i] > l2_high and c[i - 1] <= l2_prev
                if not np.isnan(l2_prev) else False)
        stL = 0 if stL == 3 else stL
        if up[i] and stL == 0: stL = 1
        if L2[i] and stL == 1: stL = 2
        if conf and stL == 2: stL = 3
        sigL[i] = stL == 3
        # short
        if S2[i]:
            s2_prev, s2_low = s2_low, l[i]
        else:
            s2_prev = s2_low
        confs = (not np.isnan(s2_low) and c[i] < s2_low and c[i - 1] >= s2_prev
                 if not np.isnan(s2_prev) else False)
        stS = 0 if stS == 3 else stS
        if dn[i] and stS == 0: stS = 1
        if S2[i] and stS == 1: stS = 2
        if confs and stS == 2: stS = 3
        sigS[i] = stS == 3
    return {"FVG First Touch Confirmed": (sigL, sigS)}


# ───────────────────────── zig-zag shared by Sweep and MSS ─────────────────────────

def zigzag_series(h, l, period, max_size, picks):
    """Replays the Pine zig-zag array bar by bar. `picks` maps name -> offset
    from the end of the flat array (Pine: array.get(zz, size - offset))."""
    n = len(h)
    out = {k: np.full(n, NAN) for k in picks}
    hh = pd.Series(h).rolling(period, min_periods=period).max().to_numpy()
    ll = pd.Series(l).rolling(period, min_periods=period).min().to_numpy()
    zz = []
    dir_ = 0
    prev_dir = None
    for i in range(n):
        ph = h[i] if (not np.isnan(hh[i]) and h[i] >= hh[i]) else NAN
        pl = l[i] if (not np.isnan(ll[i]) and l[i] <= ll[i]) else NAN
        if not np.isnan(pl) and np.isnan(ph):
            dir_ = -1
        elif not np.isnan(ph) and np.isnan(pl):
            dir_ = 1
        changed = prev_dir is not None and dir_ != prev_dir
        if not np.isnan(ph) or not np.isnan(pl):
            val = ph if dir_ == 1 else pl
            if not np.isnan(val):
                if changed or len(zz) == 0:
                    zz[0:0] = [val, float(i)]
                    if len(zz) > max_size:
                        del zz[-2:]
                elif (dir_ == 1 and val > zz[0]) or (dir_ == -1 and val < zz[0]):
                    zz[0], zz[1] = val, float(i)
        prev_dir = dir_
        s = len(zz)
        for k, off in picks.items():
            if s > off - 1:
                out[k][i] = zz[s - off]
    return out


# ───────────────────────── 3. ICT Liquidity Sweep — Prev 2 Swing Sweep ─────────────────────────

def liquidity_sweep(o, h, l, c, v):
    z = zigzag_series(h, l, 4, 14, {f"z{k}": 16 - 2 * k for k in range(1, 8)})
    z1, z2, z3, z4, z5, z6, z7 = (z[f"z{k}"] for k in range(1, 8))
    with np.errstate(invalid="ignore"):
        bear_shape = ((z2 < z3) & (z2 < z4) & (z2 < z6) & (z2 < z7) &
                      (z3 < z5) & (z3 > z4) & (z3 > z2) &
                      (z4 < z3) & (z4 < z5) & (z4 < z6) & (z4 < z7) &
                      (z5 > z3) & (z5 > z4) & (z5 > z6) & (z5 < z7) &
                      (z6 < z5) & (z6 < z7) & (z6 > z4) &
                      (z7 > z2) & (z7 > z3) & (z7 > z4) & (z7 > z5) & (z7 > z6))
        bull_shape = ((z2 > z3) & (z2 > z4) & (z2 > z6) & (z2 > z7) &
                      (z3 > z5) & (z3 < z4) & (z3 < z2) &
                      (z4 > z3) & (z4 > z5) & (z4 > z6) & (z4 > z7) &
                      (z5 < z3) & (z5 < z4) & (z5 < z6) & (z5 > z7) &
                      (z6 > z5) & (z6 > z7) & (z6 < z4) &
                      (z7 < z2) & (z7 < z3) & (z7 < z4) & (z7 < z5) & (z7 < z6))
    short = bear_shape & cross_over(z1, z5)
    long = bull_shape & cross_under(z1, z5)
    return {"Liquidity Sweep": (long, short)}


# ───────────────────────── 4. ICT Retracement to Order Block (signal 3) ─────────────────────────

def retracement_ob(o, h, l, c, v, period=30):
    n = len(c)
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum = np.abs(c - o) / rng > 0.66
    a = atr(h, l, c, 14)
    atr_ok = rng >= a
    hh = np.roll(pd.Series(h).rolling(period).max().to_numpy(), 1); hh[0] = NAN
    lll = np.roll(pd.Series(l).rolling(period).min().to_numpy(), 1); lll[0] = NAN
    pL = (o > c) & (l < lll) & momentum & atr_ok
    pS = (o < c) & (h > hh) & momentum & atr_ok

    # exrem: alternate long/short primaries
    pLs = np.zeros(n, bool); pSs = np.zeros(n, bool)
    sig = 0
    for i in range(n):
        new = 1 if pL[i] else (-1 if pS[i] else sig)
        if new != sig:
            pLs[i] = new == 1; pSs[i] = new == -1
        sig = new

    sigL = np.zeros(n, bool); sigS = np.zeros(n, bool)
    hiL = loS = NAN
    st2L = st2S = st3L = st3S = 0
    prev_hiL = prev_loS = NAN
    for i in range(n):
        if pLs[i]: hiL = h[i]
        if pSs[i]: loS = l[i]
        secL = (o[i] < c[i]) and not np.isnan(hiL) and c[i] > hiL and atr_ok[i]
        secS = (o[i] > c[i]) and not np.isnan(loS) and c[i] < loS and atr_ok[i]
        # trigger(primary, secondary)
        st2L = 0 if st2L == 2 else st2L
        if pLs[i] and st2L == 0: st2L = 1
        if secL and st2L == 1: st2L = 2
        st2S = 0 if st2S == 2 else st2S
        if pSs[i] and st2S == 0: st2S = 1
        if secS and st2S == 1: st2S = 2
        secLs, secSs = st2L == 2, st2S == 2
        # tertiary: retrace into the order-block candle
        terL = i > 0 and _xo(-l[i - 1], -prev_hiL, -l[i], -hiL)
        terS = i > 0 and _xo(h[i - 1], prev_loS, h[i], loS)
        st3L = 0 if st3L == 2 else st3L
        if secLs and st3L == 0: st3L = 1
        if terL and st3L == 1: st3L = 2
        st3S = 0 if st3S == 2 else st3S
        if secSs and st3S == 0: st3S = 1
        if terS and st3S == 1: st3S = 2
        sigL[i], sigS[i] = st3L == 2, st3S == 2
        prev_hiL, prev_loS = hiL, loS
    return {"Retracement to Order Block": (sigL, sigS)}


# ───────────────────────── 5. ICT Market Structure Shift ─────────────────────────

def mss(o, h, l, c, v, pct=0.7):
    z = zigzag_series(h, l, 4, 10, {"z1": 8, "z2": 6, "z3": 4})
    z1, z2, z3 = z["z1"], z["z2"], z["z3"]
    n = len(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        filt = np.abs((z3 - z1) / z3 * 100) > pct
        bear = ((z1 > z2) & (z1 > z3) & (z3 > z2) & (c < z1) & (c < z2) & (c < z3)
                & cross_under(c, z2) & filt)
        bull = ((z1 < z2) & (z1 < z3) & (z3 < z2) & (c > z1) & (c > z2) & (c > z3)
                & cross_over(c, z2) & filt)
    retL = np.zeros(n, bool); retS = np.zeros(n, bool)
    pL = pS = NAN; prevL = prevS = NAN
    stL = stS = 0
    for i in range(n):
        if bear[i]: pS = z1[i] - (z1[i] - z2[i]) / 3
        if bull[i]: pL = z1[i] + (z2[i] - z1[i]) / 3
        cS = i > 0 and _xo(h[i - 1], prevS, h[i], pS)
        cL = i > 0 and _xo(-l[i - 1], -prevL, -l[i], -pL)
        stS = 0 if stS == 2 else stS
        if bear[i] and stS == 0: stS = 1
        if cS and stS == 1: stS = 2
        stL = 0 if stL == 2 else stL
        if bull[i] and stL == 0: stL = 1
        if cL and stL == 1: stL = 2
        retS[i], retL[i] = stS == 2, stL == 2
        prevL, prevS = pL, pS
    return {"MSS Break": (bull, bear), "MSS Retracement": (retL, retS)}


# ───────────────────────── 6. ICT Liquidity Void Fill ─────────────────────────

def liquidity_void_fill(o, h, l, c, v, fib=0.5, body_pct=0.80):
    n = len(c)
    rng = h - l
    a = atr(h, l, c, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        big = (rng > a) & (np.abs(c - o) / rng >= body_pct)
    c1L = (o < c) & big
    c1S = (o > c) & big
    sigL = np.zeros(n, bool); sigS = np.zeros(n, bool)
    hL = lL = hS = lS = NAN
    prevFL = prevFS = NAN
    stL = stS = 0
    for i in range(n):
        if c1S[i]: hS, lS = h[i], l[i]
        if c1L[i]: hL, lL = h[i], l[i]
        fS = hS - (hS - lS) * fib        # fibonacci(false, ...) from the high
        fL = lL + (hL - lL) * fib        # fibonacci(true, ...) from the low
        cS = i > 0 and _xo(h[i - 1], prevFS, h[i], fS)
        cL = i > 0 and _xo(-l[i - 1], -prevFL, -l[i], -fL)
        stS = 0 if stS == 2 else stS
        if c1S[i] and stS == 0: stS = 1
        if (not c1S[i]) and cS and stS == 1: stS = 2
        stL = 0 if stL == 2 else stL
        if c1L[i] and stL == 0: stL = 1
        if (not c1L[i]) and cL and stL == 1: stL = 2
        sigS[i], sigL[i] = stS == 2, stL == 2
        prevFL, prevFS = fL, fS
    return {"Liquidity Void Fill": (sigL, sigS)}


# ───────────────────────── 7. Ankush Bajaj Momentum Investing ─────────────────────────

def ankush_bajaj(o, h, l, c, v, pchg=8.0, vol_mult=5.0):
    c250 = np.roll(c, 250); c250[:250] = NAN
    ch = (c - c250) / c * 100
    r = rsi(c, 14); m = mfi(h, l, c, v, 14); cc = cci(h, l, c, 20)
    vma = ema(v, 20) * vol_mult
    with np.errstate(invalid="ignore"):
        long = (ch > pchg) & (r > 60) & (m > 60) & (cc > 100) & (v > vma)
        short = (ch < -pchg) & (r < 40) & (m < 40) & (cc < -100) & (v > vma)
    return {"Ankush Bajaj Momentum": (long, short)}


# ───────────────────────── 8. RSI Directional Momentum — Continuous Break ─────────────────────────

def rsi_directional(o, h, l, c, v):
    n = len(c)
    r = rsi(c, 14); rm = sma(r, 14)
    sigL = np.zeros(n, bool); sigS = np.zeros(n, bool)
    state = 0
    s1_low = l1_high = NAN
    for i in range(1, n):
        if np.isnan(rm[i]) or np.isnan(r[i - 1]):
            continue
        state = 0 if state in (3, 6) else state
        short_1 = (r[i] < 40 and r[i - 1] >= 40) and r[i] < rm[i] and (state == 0 or 4 <= state <= 5)
        if short_1: s1_low = l[i]
        short_2 = r[i] > 40
        short_3 = (not np.isnan(s1_low)) and r[i] < 40 and r[i] < rm[i] and c[i] < s1_low
        long_1 = (r[i] > 60 and r[i - 1] <= 60) and r[i] > rm[i] and (state == 0 or 1 <= state <= 2)
        if long_1: l1_high = h[i]
        long_2 = r[i] < 60
        long_3 = (not np.isnan(l1_high)) and r[i] > 60 and r[i] > rm[i] and c[i] > l1_high
        if short_1: state = 1
        elif short_2 and state == 1: state = 2
        elif short_3 and state == 2: state = 3
        elif long_1: state = 4
        elif long_2 and state == 4: state = 5
        elif long_3 and state == 5: state = 6
        sigS[i], sigL[i] = state == 3, state == 6
    return {"RSI Directional Momentum": (sigL, sigS)}


# ───────────────────────── 9. Sideways Market Skipper (as a strategy and as a filter) ─────────────────────────

def sideways_skipper(o, h, l, c, v, range_mult=1.5):
    n = len(c)
    st, _ = supertrend(h, l, c, 3.0, 10)
    a14 = rma(true_range(h, l, c), 14)
    a_sma = sma(a14, 14)
    rng = h - l
    atr_ok = rng > a14
    xo = cross_over(c, st); xu = cross_under(c, st)
    sigL = np.zeros(n, bool); sigS = np.zeros(n, bool)
    top = bot = NAN
    stL = stS = 0
    for i in range(n):
        if xu[i]: bot, top = c[i] - a14[i] * range_mult, NAN
        if xo[i]: top, bot = c[i] + a14[i] * range_mult, NAN
        exp_ok = a14[i] > a_sma[i] if not np.isnan(a_sma[i]) else False
        brS = (not np.isnan(bot)) and c[i] < bot and o[i] > c[i] and exp_ok and atr_ok[i]
        brL = (not np.isnan(top)) and c[i] > top and o[i] < c[i] and exp_ok and atr_ok[i]
        stS = 0 if stS == 2 else stS
        if xu[i] and stS == 0: stS = 1
        if brS and stS == 1: stS = 2
        stL = 0 if stL == 2 else stL
        if xo[i] and stL == 0: stL = 1
        if brL and stL == 1: stL = 2
        sigS[i], sigL[i] = stS == 2, stL == 2
    return {"Sideways Market Skipper": (sigL, sigS)}


def skipper_regime(o, h, l, c, v):
    """Trend filter from the Skipper logic: Supertrend direction + expanding ATR."""
    st, dr = supertrend(h, l, c, 3.0, 10)
    a14 = rma(true_range(h, l, c), 14)
    with np.errstate(invalid="ignore"):
        exp_ok = a14 > sma(a14, 14)
    return (dr == -1) & exp_ok, (dr == 1) & exp_ok


STRATEGIES = [displacement, fvg, liquidity_sweep, retracement_ob, mss,
              liquidity_void_fill, ankush_bajaj, rsi_directional, sideways_skipper]


def all_signals(o, h, l, c, v):
    out = {}
    for f in STRATEGIES:
        out.update(f(o, h, l, c, v))
    return out
