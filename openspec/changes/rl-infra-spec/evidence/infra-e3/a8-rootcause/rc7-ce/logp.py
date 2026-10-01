import torch
R='/home/michael/work/infra-e3-gpu/a8rc/'
a=torch.load(R+'out-a8rc-rc3-20261001/work/packed/dA1_trace_s2_dp0.pt',weights_only=False)['small']
b=torch.load(R+'out-a8rc-rc5a-20261001/work/packed/gA1d_trace_s2_dp0.pt',weights_only=False)['small']
tot=dif=0; mx=0; ulps=[]
rows=[]
for mb in sorted(a):
    x=a[mb]['log_probs'][0].float(); y=b[mb]['log_probs'][0].float()
    d=(x!=y); tot+=x.numel(); dif+=int(d.sum())
    if d.any():
        mx=max(mx,float((x-y).abs().max()))
        # relative diff in ulp of logp magnitude: use bit distance
        xi=x.view(torch.int32).to(torch.int64); yi=y.view(torch.int32).to(torch.int64)
        ulps+= (xi-yi).abs()[d].tolist()
    rows.append((mb,x.numel(),int(d.sum()),float((x-y).abs().max()), float(x[d].abs().min()) if d.any() else None))
print('tokens',tot,'differing',dif,'max abs diff',mx)
import statistics
print('bit-distance (ulps in fp32 repr of logp): median',statistics.median(ulps),'max',max(ulps),'frac<=2ulp',sum(u<=2 for u in ulps)/len(ulps))
for r in rows[:6]: print(r)
# are old logp the same in A8's data? train_rollout_logprob_abs_diff
print({mb:(a[mb]['loss_metrics']['train_rollout_logprob_abs_diff'],b[mb]['loss_metrics']['train_rollout_logprob_abs_diff']) for mb in (8,9,10)})
