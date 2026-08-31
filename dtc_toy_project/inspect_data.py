import json

with open('data/all_cases.json') as f:
    raw = json.load(f)

print('Top-level type:', type(raw))
if isinstance(raw, dict):
    print('Top-level keys:', list(raw.keys())[:10])
    for k, v in list(raw.items())[:5]:
        print('  key=' + repr(k) + '  value type=' + str(type(v)))
elif isinstance(raw, list):
    print('List length:', len(raw))
    print('First element type:', type(raw[0]))
    if isinstance(raw[0], dict):
        print('First element keys:', list(raw[0].keys()))
    else:
        print('First element value:', raw[0])