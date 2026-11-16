"""Print a compressed-size table (bytes) from data/results.jsonl."""
import json, sys
rows = {}
for l in open('data/results.jsonl'):
    r = json.loads(l)
    if r.get('skipped'): continue
    rows[(r['method'], r['file'])] = r
cols = sys.argv[1].split(',') if len(sys.argv) > 1 else ['xz -9e', 'brotli -q11', 'zstd -22 --long', 'ours-cm', 'ours-columnar+xz', 'ours-columnar+cm']
filt = sys.argv[2] if len(sys.argv) > 2 else ''
files = sorted({f for (_, f) in rows if filt in f})
print(f"{'file':42}" + ''.join(f"{c[:16]:>17}" for c in cols))
for f in files:
    line = f"{f[:42]:42}"
    for c in cols:
        r = rows.get((c, f))
        line += f"{(r['compressed'] if r else '-'):>17}"
    print(line)
