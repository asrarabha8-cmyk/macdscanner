"""Backtest of the 'last strong candle' control-flip formula (MetaStock-style).
Rules are a literal translation. Pass/fail criteria are fixed BEFORE seeing results.
Signal on bar close -> fill at NEXT bar open -> cost per side.
Usage: python research/candle_control_bt.py [--csv-dir DIR]
CSV format: <SYMBOL>_<1d|1h>.csv with columns date,open,high,low,close
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

SYMBOLS = ["SPY","QQQ","IWM","DIA","XLK","SMH","AAPL","MSFT","NVDA","AMZN",
           "GOOGL","META","TSLA","AMD","JPM","XOM"]
COSTS_BPS = [0, 5, 10]          # per side
MAIN_COST = 5
SA_SYMBOLS = ["^TASI.SR","2222.SR","1120.SR","1180.SR","2010.SR","7010.SR","1211.SR",
              "1150.SR","1010.SR","2280.SR","5110.SR","2082.SR"]
SA_COST = 15                    # 0.3% round trip, same estimate as research/dip_sa.py
SPLIT = 0.7                     # first 70% = in-sample, last 30% = out-of-sample
OUT = os.path.dirname(os.path.abspath(__file__))

def signals(df, thr=0.55):
    o,h,l,c = df.open, df.high, df.low, df.close
    pc = c.shift(1)
    ratio = (c-o).abs() / (h-l).clip(lower=1e-6)
    bull = (c>o) & (ratio>=thr)
    bear = (c<o) & (ratio>=thr)
    bull_low  = l.where(bull).ffill().shift(1)
    bear_high = h.where(bear).ffill().shift(1)
    buy_b  = (c>bear_high) & (pc<=bear_high)
    sell_b = (c<bull_low)  & (pc>=bull_low)
    broken = bear_high.where(buy_b).ffill()
    main = pd.Series(np.where(buy_b,1.0,np.where(sell_b,-1.0,np.nan)), index=df.index).ffill()
    false_b = (main==1) & (c<broken) & (pc>=broken)
    last = pd.Series(np.where(buy_b,1.0,np.where(sell_b|false_b,-1.0,np.nan)), index=df.index).ffill().shift(1)
    buy  = buy_b & (last.fillna(0)!=1)
    sell = (sell_b|false_b) & (last.fillna(0)!=-1)
    return pd.Series(np.where(buy,1,np.where(sell,-1,0)), index=df.index)

def run(df, sig, mode, bps):
    """mode: long = long/flat, ls = long/short. Position starts at open AFTER the signal."""
    tgt = sig.replace(0,np.nan).ffill().fillna(0)
    if mode=="long": tgt = tgt.clip(lower=0)
    pos = tgt.shift(1).fillna(0)                       # held from this bar's open
    r_oo = df.open.shift(-1)/df.open - 1               # open -> next open
    chg = pos.diff().abs().fillna(pos.abs())
    ret = (pos*r_oo - chg*bps/1e4).iloc[:-1].fillna(0)
    trades=[]; cur=0; entry=None
    p=pos.values; op=df.open.values
    for i in range(len(p)):
        if p[i]!=cur:
            if cur!=0:
                trades.append(cur*(op[i]/entry-1)-2*bps/1e4)
            cur=p[i]; entry=op[i] if cur!=0 else None
    return ret, pos.iloc[:-1], np.array(trades)

def stats(ret, pos, trades, per_year):
    if len(ret)<2: return None
    eq=(1+ret).cumprod(); yrs=len(ret)/per_year
    cagr=eq.iloc[-1]**(1/yrs)-1 if eq.iloc[-1]>0 else -1
    dd=(eq/eq.cummax()-1).min()
    sh=ret.mean()/ret.std()*np.sqrt(per_year) if ret.std()>0 else 0
    w=trades[trades>0]; ls=trades[trades<=0]
    pf=(w.sum()/-ls.sum()) if len(ls) and ls.sum()<0 else float("inf")
    return dict(trades=int(len(trades)), win=float((trades>0).mean()) if len(trades) else 0,
                pf=float(pf), avg=float(trades.mean()) if len(trades) else 0,
                cagr=float(cagr), maxdd=float(dd), sharpe=float(sh),
                exposure=float((pos!=0).mean()))

def bh(df, per_year):
    r=(df.open.shift(-1)/df.open-1).iloc[:-1].fillna(0)
    return stats(r, pd.Series(1,index=r.index), np.array([df.open.iloc[-1]/df.open.iloc[0]-1]), per_year)

def load(sym, tf, csv_dir):
    if csv_dir:
        p=os.path.join(csv_dir,f"{sym}_{tf}.csv")
        if not os.path.exists(p): return None
        return pd.read_csv(p,parse_dates=["date"],index_col="date")[["open","high","low","close"]].dropna()
    import yfinance as yf
    kw=dict(period="max",interval=tf) if tf in("1d","1wk") else dict(period="729d",interval="1h")
    d=yf.download(sym,auto_adjust=True,progress=False,**kw)
    if d is None or d.empty: return None
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d.rename(columns=str.lower)[["open","high","low","close"]].dropna()
    return d[d.index>="2005-01-01"] if tf in("1d","1wk") else d

def verdict(rows):
    """Fixed before seeing results. Judged on OOS, main cost, per timeframe+mode."""
    v=[r for r in rows if r["oos"] and r["oos"]["trades"]>0]
    if not v: return dict(passed=False,n=0)
    tr=sum(r["oos"]["trades"] for r in v)
    gw=np.median([min(r["oos"]["pf"],10) for r in v])
    beat=np.mean([r["oos"]["sharpe"]>r["oos_bh"]["sharpe"] for r in v])
    pos=np.mean([r["oos"]["pf"]>1 for r in v])
    return dict(n=len(v),trades=int(tr),median_pf=float(gw),beat_bh_sharpe=float(beat),pf_gt1=float(pos),
                passed=bool(tr>=100 and gw>=1.3 and beat>=0.6 and pos>=0.7))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv-dir"); ap.add_argument("--out",default=OUT); ap.add_argument("--tf",default=""); ap.add_argument("--market",default="us")
    a=ap.parse_args()
    global SYMBOLS, COSTS_BPS, MAIN_COST
    minbars=300; tag=""
    if a.market=="sa":
        SYMBOLS=SA_SYMBOLS; MAIN_COST=SA_COST; COSTS_BPS=[0,5,SA_COST]; minbars=200; tag="_sa"
    tag+= "_"+a.tf if a.tf else ""
    res={}; md=["# Candle Control Backtest","",
      f"Next-open fills. Market {a.market}, main cost {MAIN_COST} bps per side. PASS (out-of-sample, main cost): >=100 trades, median PF>=1.3, Sharpe beats buy&hold on >=60% of symbols, PF>1 on >=70%.",""]
    for tf,py in (("1d",252),("1h",252*7),("1wk",52)):
        if a.tf and tf!=a.tf: continue
        for mode in ("long","ls"):
            rows=[]
            for s in SYMBOLS:
                try: df=load(s,tf,a.csv_dir)
                except Exception as e: print(s,tf,e,file=sys.stderr); df=None
                if df is None or len(df)<minbars: continue
                sig=signals(df); cut=int(len(df)*SPLIT); row=dict(sym=s,bars=len(df),start=str(df.index[0])[:10],end=str(df.index[-1])[:10])
                for b in COSTS_BPS:
                    ret,pos,tr=run(df,sig,mode,b)
                    row[f"all_{b}"]=stats(ret,pos,tr,py)
                # OOS: signals computed on full history, sim restricted to last 30%
                d2=df.iloc[cut:]; ret,pos,tr=run(d2,sig.iloc[cut:],mode,MAIN_COST)
                row["oos"]=stats(ret,pos,tr,py); row["oos_bh"]=bh(d2,py); row["all_bh"]=bh(df,py)
                rows.append(row)
            k=f"{tf}_{mode}"; v=verdict(rows); res[k]=dict(verdict=v,rows=rows)
            md+= [f"## {k} — {'PASS' if v.get('passed') else 'FAIL'}", f"`{json.dumps(v)}`","",
                  "|sym|trades|win|PF|CAGR|MaxDD|Sharpe|BH CAGR|BH MaxDD|BH Sharpe|OOS PF|OOS Sharpe|OOS BH Sharpe|","|"+"---|"*13]
            for r in rows:
                x=r[f"all_{MAIN_COST}"]; b=r["all_bh"]; o=r["oos"]; ob=r["oos_bh"]
                md.append(f"|{r['sym']}|{x['trades']}|{x['win']:.0%}|{x['pf']:.2f}|{x['cagr']:.1%}|{x['maxdd']:.0%}|{x['sharpe']:.2f}|{b['cagr']:.1%}|{b['maxdd']:.0%}|{b['sharpe']:.2f}|{o['pf']:.2f}|{o['sharpe']:.2f}|{ob['sharpe']:.2f}|")
            md.append("")
    open(os.path.join(a.out,f"candle_control_results{tag}.md"),"w").write("\n".join(md))
    json.dump(res,open(os.path.join(a.out,f"candle_control_results{tag}.json"),"w"),indent=1)
    print("\n".join(md))

if __name__=="__main__": main()
