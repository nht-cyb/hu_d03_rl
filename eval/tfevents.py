"""Minimal TensorBoard event-file reader (scalars only), no tensorboard needed."""
import struct, sys, json
from collections import defaultdict

def varint(b, i):
  r = s = 0
  while True:
    c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
    if c < 0x80: return r, i

def fields(b):
  i = 0
  while i < len(b):
    key, i = varint(b, i); f, t = key >> 3, key & 7
    if t == 0: v, i = varint(b, i)
    elif t == 1: v = b[i:i+8]; i += 8
    elif t == 5: v = b[i:i+4]; i += 4
    elif t == 2: n, i = varint(b, i); v = b[i:i+n]; i += n
    else: raise ValueError(t)
    yield f, t, v

def read(path):
  out = defaultdict(dict); data = open(path, "rb").read(); i = 0
  while i < len(data):
    n = struct.unpack("<Q", data[i:i+8])[0]; rec = data[i+12:i+12+n]; i += 16 + n
    step, summ = 0, None
    for f, t, v in fields(rec):
      if f == 2: step = v
      elif f == 5: summ = v
    if summ is None: continue
    for f, t, v in fields(summ):
      if f != 1: continue
      tag = val = None
      for g, u, w in fields(v):
        if g == 1: tag = w.decode()
        elif g == 2: val = struct.unpack("<f", w)[0]
      if tag is not None and val is not None: out[tag][step] = val
  return out

if __name__ == "__main__":
  d = read(sys.argv[1])
  for k in sorted(d): print(f"{k:60s} {len(d[k]):6d} steps {min(d[k])}-{max(d[k])}")
