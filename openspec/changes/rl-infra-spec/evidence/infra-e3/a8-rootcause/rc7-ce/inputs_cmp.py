import torch, sys
R='/home/michael/work/infra-e3-gpu/a8rc/'
def L(p): return torch.load(R+p, weights_only=False)
def eqv(a,b):
    if torch.is_tensor(a) and torch.is_tensor(b): return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)): return len(a)==len(b) and all(eqv(x,y) for x,y in zip(a,b))
    if isinstance(a,dict) and isinstance(b,dict): return a.keys()==b.keys() and all(eqv(a[k],b[k]) for k in a)
    return a==b
def small_cmp(name, a, b):
    sa,sb=a.get('small') or {},b.get('small') or {}
    print(f'== {name}: mbs {len(sa)} vs {len(sb)}')
    fields={}
    for mb in sorted(set(sa)&set(sb)):
        for k in sorted(set(sa[mb])|set(sb[mb])):
            fields.setdefault(k,[]).append(eqv(sa[mb].get(k),sb[mb].get(k)))
    for k,v in fields.items(): print(f'  small.{k}: equal in {sum(v)}/{len(v)} micro batches')
    ra,rb=a['records'],b['records']
    keys=sorted(set(ra)&set(rb))
    def cat(k):
        for c in ('<root-input>','|fwd|','|bwd_out|','|bwd_in|'):
            if c in k: return ('fo ' if k.startswith('fo') else '')+c
        return 'other'
    stat={}
    for k in keys:
        s=stat.setdefault(cat(k),[0,0]); s[1]+=1; s[0]+= (ra[k]['bits']==rb[k]['bits'])
    print('  records (common keys %d; only-a %d; only-b %d):'%(len(keys),len(set(ra)-set(rb)),len(set(rb)-set(ra))), {k:f'{v[0]}/{v[1]} equal' for k,v in stat.items()})
    firstdiff=[k for k in keys if ra[k]['bits']!=rb[k]['bits']][:5]
    print('  first differing record keys:',firstdiff)
# RC-3 dA1 (ref class) vs RC-5a gA1d (class X): deep probes, DP1, steps 1 and 2
for s in (2,):
    a=L(f'out-a8rc-rc3-20261001/work/packed/dA1_trace_s{s}_dp0.pt')
    try: b=L(f'out-a8rc-rc5a-20261001/work/packed/gA1d_trace_s{s}_dp0.pt')
    except Exception as e: print('no rc5a s',s,e); continue
    small_cmp(f'DP1 step{s}: RC-3 dA1 (ref class) vs RC-5a gA1d (X)',a,b)
# RC-2 rcA1 (ref) vs RC-5a gA1d (X): module-hook records only
a=L('out-a8rc-rc2-20261001/work/packed/rcA1_trace_s2_dp0.pt'); b=L('out-a8rc-rc5a-20261001/work/packed/gA1d_trace_s2_dp0.pt')
small_cmp('DP1 step2: RC-2 rcA1 (ref) vs RC-5a gA1d (X)',a,b)
# DP2: RC-2 rcA2 (B1 class) vs RC-5a gA2d (X) by rank
for dp in (0,1):
    a=L(f'out-a8rc-rc2-20261001/work/packed/rcA2_trace_s2_dp{dp}.pt'); b=L(f'out-a8rc-rc5a-20261001/work/packed/gA2d_trace_s2_dp{dp}.pt')
    small_cmp(f'DP2 rank{dp} step2: RC-2 rcA2 (B1 class) vs RC-5a gA2d (X)',a,b)
