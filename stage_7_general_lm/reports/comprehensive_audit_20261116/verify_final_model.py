"""Load the trusted local final checkpoint and check inference, without updates."""
import json
import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from stage_7_general_lm.common import load_checkpoint
from stage_7_general_lm.model import LanguageModel

torch.set_num_threads(2)
torch.manual_seed(20261116)
p = ROOT / 'stage_7_general_lm/artifacts/checkpoints/2b_1150579/review_u122071_positions_2000011264.pt'
c = load_checkpoint(p)
print('Checkpoint loaded; fields:',list(c),flush=True)
cfg = c.get('config',c.get('cfg'))
assert cfg is not None
model = LanguageModel(cfg['model'])
model.load_state_dict(c['model'],strict=True)
model.eval()
count = sum(x.numel() for x in model.parameters())
assert count == 122702592
assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
assert all(torch.isfinite(x).all().item() for x in model.parameters())
optimizer = c['optimizer']
steps = []
for state in optimizer['state'].values():
    steps.append(int(state['step']))
    for key in ['exp_avg','exp_avg_sq']:
        assert torch.isfinite(state[key]).all().item()
assert set(steps) == {122071}
tokenizer = Tokenizer.from_file(str(ROOT / 'stage_7_general_lm/artifacts/native_tokenizer/tokenizer.json'))
prompt = 'Plants use sunlight to make food through photosynthesis. During this process,'
ids = torch.tensor([tokenizer.encode(prompt,add_special_tokens=False).ids],dtype=torch.long)
with torch.no_grad():
    full,_ = model(ids)
    prefix,_,cache = model(ids[:,:-1],use_cache=True)
    last,_,_ = model(ids[:,-1:],past=cache,use_cache=True)
    assert torch.isfinite(full).all().item()
    error = (full[:,-1:]-last).abs().max().item()
    torch.testing.assert_close(full[:,-1:],last,atol=3e-5,rtol=1e-5)
    generated = model.greedy(ids,new_tokens=24,eos_id=2)
result = dict(status='PASS',device='cpu',torch_version=str(torch.__version__),
              checkpoint_bytes=p.stat().st_size,unique_parameters=count,
              strict_state_load=True,tied_embedding=True,all_parameters_finite=True,
              optimizer_states=len(steps),optimizer_step=steps[0],optimizer_moments_finite=True,
              prompt_tokens=ids.size(1),cached_logits_max_abs_error=error,
              prompt=prompt,greedy_continuation=tokenizer.decode(generated[0,ids.size(1):].tolist()),
              scope='Read-only CPU inference; no optimizer updates, no cross-runtime resume claim')
(Path(__file__).parent / 'final_model_cpu_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
