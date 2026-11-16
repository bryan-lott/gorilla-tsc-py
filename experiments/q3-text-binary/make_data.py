"""Derive binary test files from the public bench data (run from data/)."""
import struct, calendar, sqlite3, os
B = '../../../bench-data/'
vals = [float(l.rsplit(',', 1)[1]) for l in open(B + 'city_temperature.csv')]
open('city_temp.f64', 'wb').write(struct.pack('<%dd' % len(vals), *vals))
open('city_temp.f32', 'wb').write(struct.pack('<%df' % len(vals), *vals))
out = bytearray(); rows = []
with open(B + 'Stocks-Germany-sample.txt') as f:
    for i, l in enumerate(f):
        if i >= 1000000: break
        d, t, p, v = l.rstrip('\n').split(',')
        m, dd, y = map(int, d.split('/')); hh, mm, ss = map(int, t.split(':'))
        ts = calendar.timegm((y, m, dd, hh, mm, ss))
        out += struct.pack('<qdi', ts, float(p), int(v)); rows.append((ts, float(p), int(v)))
open('stocks_records.bin', 'wb').write(out)   # 20-byte records
if os.path.exists('bench.sqlite'): os.remove('bench.sqlite')
db = sqlite3.connect('bench.sqlite')
db.execute('create table trades(id integer primary key, ts integer, price real, volume integer)')
db.executemany('insert into trades(ts,price,volume) values (?,?,?)', rows[:300000])
db.execute('create table temps(region text, country text, temp real)')
db.executemany('insert into temps values (?,?,?)', (l.rstrip('\n').split(',') for l in list(open(B + 'city_temperature.csv'))[:300000]))
db.commit(); db.execute('vacuum'); db.close()
