import sys; sys.path.insert(0,'/home/michael/work/infra-e3/tools/probes/e3_reshard')
import torch
from compare_rc import _load, flat, rel
R='/home/michael/work/infra-e3-gpu/'
known={'A8_A2_s2(class B1)':R+'b3a8r/out/work/packed/A2_s2.pt','RC4_eA2_s2(good)':R+'a8rc/out-a8rc-rc4-20261001/work/packed/eA2_s2.pt','RC5a_gA2_s2(class X)':R+'a8rc/out-a8rc-rc5a-20261001/work/packed/gA2_s2.pt','A8_A1_s2(ref DP1)':R+'b3a8r/out/work/packed/A1_s2.pt'}
ks={k:flat(_load(v),'exp_avg') for k,v in known.items()}
for i in (1,2,3,'cf1','cf2'):
    m=flat(_load(R+f'a8rc/out-a8rc-ce{i}-20261001/work/packed/gA2e_s2.pt' if isinstance(i,int) else R+f'a8rc/out-a8rc-{i}-20261001/work/packed/cA2e_s2.pt'),'exp_avg')
    print(i,{k:('BITWISE' if torch.equal(m,v) else f'{rel(v,m):.3e}') for k,v in ks.items()})
