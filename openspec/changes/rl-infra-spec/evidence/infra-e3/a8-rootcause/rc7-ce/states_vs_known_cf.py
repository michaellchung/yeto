import sys; sys.path.insert(0,'/home/michael/work/infra-e3/tools/probes/e3_reshard')
import torch
from compare_rc import _load, flat, rel
R='/home/michael/work/infra-e3-gpu/'
known={'A8_A2_s2(B1)':R+'b3a8r/out/work/packed/A2_s2.pt','RC4_eA2_s2(good)':R+'a8rc/out-a8rc-rc4-20261001/work/packed/eA2_s2.pt','RC5a_gA2_s2(X)':R+'a8rc/out-a8rc-rc5a-20261001/work/packed/gA2_s2.pt','A8_A1_s2(ref)':R+'b3a8r/out/work/packed/A1_s2.pt'}
ks={k:flat(_load(v),'exp_avg') for k,v in known.items()}
for name,p in (('cf1','a8rc/out-a8rc-cf1-20261001/work/packed/cA2e_s2.pt'),('cf2','a8rc/out-a8rc-cf2-20261001/work/packed/cA2e_s2.pt')):
    m=flat(_load(R+p),'exp_avg')
    print(name,{k:('BITWISE' if torch.equal(m,v) else f'{rel(v,m):.3e}') for k,v in ks.items()})
