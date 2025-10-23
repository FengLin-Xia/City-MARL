import json
with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print('Config valid')
print(f'Has scheduler: {"scheduler" in cfg}')
params = cfg.get('scheduler',{}).get('params',{})
print(f'Has phases: {"phases" in params}')
phases = params.get('phases',[])
print(f'Phases count: {len(phases)}')
if phases:
    print(f'Phase 0: {phases[0]}')




