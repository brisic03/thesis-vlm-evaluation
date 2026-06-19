import pandas as pd, numpy as np, math

REPO="/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
T=pd.read_csv(f"{REPO}/results/step2/step2b_tinyllava_numeric.csv")
M=pd.read_csv(f"{REPO}/results/step2/step2b_mobilevlm_numeric.csv")

def bar(n, total, width=40, ch="#"):
    if total==0: return ""
    k=int(round(width*n/total))
    return ch*k

def hist_block(title, counts_labels, total):
    print(title)
    maxn=max((c for _,c in counts_labels), default=1)
    for lab,c in counts_labels:
        pct=100*c/total if total else 0
        k=int(round(40*c/maxn))
        print(f"  {lab:>10} | {'█'*k:<40} {c:4d}  ({pct:4.1f}%)")
    print()

for name,d in [("TinyLLaVA",T),("MobileVLM",M)]:
    print("="*72)
    print(f"  {name} — VisDrone numeric counting  (545 questions total)")
    print("="*72)
    n_total=len(d)
    ok=d[d.parse_ok].copy()
    n_ok=len(ok)
    n_refuse=n_total-n_ok
    print(f"  answered with a number : {n_ok}/{n_total}  ({100*n_ok/n_total:.1f}%)")
    print(f"  refused / no number    : {n_refuse}/{n_total}  ({100*n_refuse/n_total:.1f}%)")
    err=ok.pred_count - ok.gt_count          # signed
    ae=err.abs()
    print(f"  MAE={ae.mean():.2f}  RMSE={math.sqrt((err**2).mean()):.2f}  median|err|={ae.median():.0f}  bias(mean signed)={err.mean():+.2f}")
    print()

    # --- absolute error buckets ---
    buckets=[("exact (0)", (ae==0)),
             ("off by 1", (ae==1)),
             ("off by 2", (ae==2)),
             ("off by 3-5", (ae>=3)&(ae<=5)),
             ("off by 6-10",(ae>=6)&(ae<=10)),
             ("off by >10", (ae>10))]
    hist_block("  HOW FAR OFF (absolute error |guess - truth|), over answered Qs:",
               [(lab, int(mask.sum())) for lab,mask in buckets], n_ok)

    # cumulative accuracy within tolerance
    print("  WITHIN-TOLERANCE accuracy (answered Qs):")
    for tol in [0,1,2,3,5]:
        c=int((ae<=tol).sum()); print(f"     |err| <= {tol}: {c:4d}  ({100*c/n_ok:4.1f}%)  {bar(c,n_ok)}")
    print()

    # --- signed error histogram ---
    edges=[(-99,-11,"<= -11 (way under)"),(-10,-6,"-10..-6"),(-5,-3,"-5..-3"),
           (-2,-2,"-2"),(-1,-1,"-1"),(0,0,"0 exact"),(1,1,"+1"),(2,2,"+2"),
           (3,5,"+3..+5"),(6,10,"+6..+10"),(11,99,">= +11 (way over)")]
    rows=[]
    for lo,hi,lab in edges:
        c=int(((err>=lo)&(err<=hi)).sum())
        rows.append((lab,c))
    hist_block("  SIGNED error (negative = UNDER-count, positive = OVER-count):",
               rows, n_ok)

print("="*72)
print("  WHEN TinyLLaVA REFUSES vs ANSWERS — by true count")
print("="*72)
ok=T[T.parse_ok]; ref=T[~T.parse_ok]
gtbuckets=[("1-2",1,2),("3-5",3,5),("6-10",6,10),("11+",11,9999)]
print(f"  {'true count':>11} | {'answered':>8} {'refused':>8}  refusal-rate")
for lab,lo,hi in gtbuckets:
    a=((ok.gt_count>=lo)&(ok.gt_count<=hi)).sum()
    r=((ref.gt_count>=lo)&(ref.gt_count<=hi)).sum()
    tot=a+r
    rr=100*r/tot if tot else 0
    print(f"  {lab:>11} | {a:8d} {r:8d}  {rr:4.0f}%  {'░'*int(rr/2.5)}")
print()

print("="*72)
print("  WHAT NUMBERS THEY GUESS (top predicted values, answered Qs)")
print("="*72)
for name,d in [("TinyLLaVA",T),("MobileVLM",M)]:
    ok=d[d.parse_ok]
    vc=ok.pred_count.value_counts().sort_values(ascending=False).head(8)
    print(f"  {name}:  (true-count mean = {ok.gt_count.mean():.1f})")
    mx=vc.max()
    for val,c in vc.items():
        print(f"     guessed {int(val):>3} : {'█'*int(round(30*c/mx)):<30} {c:4d}  ({100*c/len(ok):4.1f}%)")
    print()
