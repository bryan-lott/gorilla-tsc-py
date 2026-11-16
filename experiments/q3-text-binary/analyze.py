"""Summaries from ../results/q3.json for the findings write-up (sizes only)."""
import json, os, collections

R = json.load(open(os.path.join(os.path.dirname(__file__), '../results/q3.json')))
C = R['compressed_bytes']
files = R['files']
kind = {m['name']: m['kind'] for m in R['methods']}
base = [m for m in kind if kind[m] == 'baseline']
ours = [m for m in kind if kind[m] == 'ours' and m != 'ours-auto']

def best(ms, f):
    c = [(C[m][f], m) for m in ms if f in C.get(m, {})]
    return min(c) if c else (None, None)

agg = collections.defaultdict(lambda: collections.Counter())
print(f"{'file':52}{'bytes':>11} {'best baseline':>28} {'best ours':>32} {'ours/base':>9}")
for f in files:
    n = f['name']
    b, bm = best(base, n)
    o, om = best(ours, n)
    print(f"{n[:52]:52}{f['bytes']:>11} {b:>11} {bm[:16]:>16} {o:>11} {om[:20]:>20} {o / b:>9.3f}")
    a = agg[f['category']]
    a['raw'] += f['bytes']; a['base'] += b; a['ours'] += o
    for m in ['xz -9e', 'zstd -19', 'brotli -q11', 'ours-cm', 'ours-auto']:
        if n in C.get(m, {}):
            a[m] += C[m][n]
print()
for cat, a in agg.items():
    print(f"{cat:14} raw {a['raw']:>11}  best-baseline {a['base']:>10}  best-ours {a['ours']:>10} ({a['ours'] / a['base']:.3f})  "
          f"xz {a['xz -9e']:>10} zstd19 {a['zstd -19']:>10} brotli {a['brotli -q11']:>10} cm {a['ours-cm']:>10} auto {a['ours-auto']:>10}")

def table(ms, cat=None, names=None):
    sel = [f for f in files if (cat is None or f['category'] == cat) and (names is None or f['name'] in names)]
    print(f"\n{'file':40}" + ''.join(f"{m[:15]:>16}" for m in ms))
    for f in sel:
        print(f"{f['name'][:40]:40}" + ''.join(f"{C.get(m, {}).get(f['name'], '-'):>16}" for m in ms))

table(['xz -9e', 'brotli -q11', 'ours-libcoder-o2', 'ours-cm'], 'text')
table(['xz -9e', 'zstd -19', 'brotli -q11', 'ours-cm', 'ours-columnar+xz', 'ours-columnar+cm'], 'numeric-text')
table(['xz -9e', 'zstd -19', 'brotli -q11', 'ours-cm', 'ours-log+xz', 'ours-log+cm'], 'log')
table(['xz -9e', 'xz -9e --x86', 'brotli -q11', 'ours-cm', 'ours-cm-stride', 'ours-cm-bcj', 'ours-struct'], 'binary')
table(['xz -9e', 'flac -8', 'ours-cm', 'ours-cm-stride', 'ours-struct', 'ours-lpc'], 'audio')
print('\nspeeds (MB/s enc/dec) on selected large files')
for n in ['enwik8 (first 10 MB)', 'Stocks-Germany-sample.txt', 'system journal (125k lines)', 'music, 60 s 16-bit stereo PCM (decoded from MP3)', 'stock ticks, 20-byte records (i64 ts, f64 price, i32 vol)']:
    print(n[:40], {m: (R['enc_mbs'][m][n], R['dec_mbs'][m][n]) for m in R['enc_mbs'] if n in R['enc_mbs'][m]})
print('\nverified:', {m['name']: m['lossless_verified'] for m in R['methods']})
print('notes:', {m: v for m, v in R['notes'].items() if m in ('ours-struct', 'ours-cm-stride', 'ours-auto')})
