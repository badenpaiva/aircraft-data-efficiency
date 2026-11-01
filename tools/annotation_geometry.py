"""Validated YOLO box/polygon geometry used by the preparation reader."""
import math

def parse_line(line, nc):
    vals = list(map(float, line.split()))
    if not vals or not all(math.isfinite(v) for v in vals):
        raise ValueError('nonfinite/empty')
    if vals[0] != int(vals[0]) or not 0 <= vals[0] < nc:
        raise ValueError('class')
    cid, xy = int(vals[0]), vals[1:]
    if any(v < 0 or v > 1 for v in xy):
        raise ValueError('range')
    if len(xy) == 4:
        x,y,w,h = xy
        if w <= 0 or h <= 0: raise ValueError('area')
        bounds = [x-w/2,y-h/2,x+w/2,y+h/2]
        if min(bounds) < -1e-6 or max(bounds) > 1+1e-6:
            raise ValueError('box outside image')
        return cid, 'box', xy, bounds
    if len(xy) < 6 or len(xy)%2: raise ValueError('arity')
    pts = list(zip(xy[::2],xy[1::2]))
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(pts,pts[1:]+pts[:1])))/2
    if area <= 1e-12: raise ValueError('polygon area')
    xs,ys = xy[::2],xy[1::2]
    return cid,'polygon',xy,[min(xs),min(ys),max(xs),max(ys)]
