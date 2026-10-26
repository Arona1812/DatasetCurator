"""Local rectangular photo seams; proposals require human review."""
from __future__ import annotations
import numpy as np
from PIL import Image


def complete_black_frame(image, min_border=8):
    """Bound content inside opposing near-black, constant outer margins.

    A weak seam against dark clothing must not leave half the frame behind.
    Two opposing margins are required; a lone dark scene edge is insufficient.
    Keep a two-pixel cushion around all visible content, including JPEG ringing.
    """
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    h, w = pixels.shape[:2]
    active = pixels.max(axis=2) > 12
    ys, xs = np.nonzero(active)
    if not len(xs):
        return None
    box = [max(0, int(xs.min())-2), max(0, int(ys.min())-2),
           min(w, int(xs.max())+3), min(h, int(ys.max())+3)]
    margins = [box[0], box[1], w-box[2], h-box[3]]
    if not ((margins[0] >= min_border and margins[2] >= min_border)
            or (margins[1] >= min_border and margins[3] >= min_border)):
        return None
    if (box[2]-box[0])*(box[3]-box[1]) < w*h*.42:
        return None
    return box


def app_photo_viewport(image):
    """Photo texture transitions between flat app header/footer backgrounds.

    Only called after app-shell detection. Keep the complete photo including
    overlays: no search for text-free or person-sized subregions here.
    """
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    h, w = pixels.shape[:2]
    gray = pixels.mean(axis=2)
    spread = gray.std(axis=1)
    delta = np.mean(np.abs(np.diff(pixels,axis=0)),axis=2)
    tops, bottoms = [], []
    for y in range(max(12,int(h*.08)), min(h-24,int(h*.90))):
        before = float(np.median(spread[y-10:y-2]))
        after = float(np.median(spread[y+2:y+14]))
        coverage = float(np.mean(delta[y-1] > 2))
        strength = float(delta[y-1].mean())
        if coverage < .60 or strength < 4:
            continue
        if y < h*.42 and before < 2 and after > 2.5 and spread[y-1] < 2:
            tops.append((strength*coverage,y))
        if y > h*.55 and before > 4 and after < 2 and spread[y] < 2:
            bottoms.append((strength*coverage,y))
    if not tops or not bottoms:
        return None
    top, bottom = max(tops)[1], max(bottoms)[1]
    if bottom-top < h*.30:
        return None
    return [0,max(0,top-1),w,min(h,bottom+1)]


def colour_inset_photo(image):
    """Find a coloured rectangular inset against a monochrome story backdrop.

    Saturation locates the domain; long seams corroborate all four boundaries.
    Ordinary clothing patches without a closed rectangle are rejected.
    """
    import cv2
    pixels = np.asarray(image.convert('RGB'),dtype=np.uint8)
    h,w = pixels.shape[:2]
    saturation = cv2.cvtColor(pixels,cv2.COLOR_RGB2HSV)[:,:,1]
    mask = cv2.morphologyEx((saturation>60).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask,8)
    options = []
    for x,y,bw,bh,area in stats[1:]:
        x,y,bw,bh,area = map(int,(x,y,bw,bh,area))
        if not .075 <= bw*bh/(w*h) <= .60 or area/(bw*bh) < .25 or min(x,y,w-x-bw,h-y-bh)<8:
            continue
        box=[x,y,x+bw,y+bh]
        support=[]
        floating=pixels.astype(np.float32)
        for side in range(4):
            vertical=side%2==0
            limit=w if vertical else h
            start,end=(y+8,y+bh-8) if vertical else (x+8,x+bw-8)
            region=floating[start:end].transpose(1,0,2) if vertical else floating[:,start:end]
            sat_region=saturation[start:end].T if vertical else saturation[:,start:end]
            gradients=np.mean(np.abs(np.diff(region,axis=0)),axis=2)
            candidates=[]
            for coordinate in range(max(2,box[side]-14),min(limit-2,box[side]+15)):
                edge=np.max(gradients[max(0,coordinate-2):coordinate+1],axis=0)
                coverage=float(np.mean(edge>8))
                outer=sat_region[max(0,coordinate-5):coordinate] if side<2 else sat_region[coordinate:coordinate+5]
                inner=sat_region[coordinate:coordinate+5] if side<2 else sat_region[max(0,coordinate-5):coordinate]
                gain=float(inner.mean())-float(outer.mean())
                if gain >= 15:
                    candidates.append((coverage*(1+float(edge.mean())/25)*min(2,gain/35),coverage,coordinate))
            if not candidates:
                support.append(0)
                continue
            _, coverage, coordinate=max(candidates)
            box[side]=coordinate
            support.append(coverage)
        if min(support)<.60 or np.mean(support)<.75:
            continue
        x1,y1,x2,y2=box
        if x2<=x1 or y2<=y1:
            continue
        ring=np.concatenate((saturation[y1:y2,x1-5:x1].ravel(),saturation[y1:y2,x2:x2+5].ravel(),
                             saturation[y1-5:y1,x1:x2].ravel(),saturation[y2:y2+5,x1:x2].ravel()))
        if float(saturation[y1:y2,x1:x2].mean())-float(ring.mean())<35:
            continue
        options.append((float(np.mean(support)),box))
    if not options:
        return None
    box=max(options,key=lambda value:value[0])[1]
    # Seam estimates on downscaled screenshots can land inside hair/clothing.
    # Keep a small exterior cushion rather than optimizing to the last pixel.
    return [max(0,box[0]-3),max(0,box[1]-3),min(w,box[2]+3),min(h,box[3]+3)]


def boundary_evidence(image, box):
    pixels = np.asarray(image,dtype=np.float32)
    h,w = pixels.shape[:2]
    x1,y1,x2,y2 = box
    scores, textures = [], []
    for axis,coordinate,start,end in ((1,x1,y1,y2),(1,x2,y1,y2),(0,y1,x1,x2),(0,y2,x1,x2)):
        length = w if axis == 1 else h
        if coordinate <= 1 or coordinate >= length-1:
            continue
        patch = pixels[start:end,max(0,coordinate-3):min(w,coordinate+3)] if axis == 1 else pixels[max(0,coordinate-3):min(h,coordinate+3),start:end].transpose(1,0,2)
        scores.append(float(np.mean(np.max(np.mean(np.abs(np.diff(patch,axis=1)),axis=2),axis=1)>10)))
        is_start_edge = coordinate == (x1 if axis == 1 else y1)
        outer_start,outer_end = (max(0,coordinate-7),max(0,coordinate-2)) if is_start_edge else (min(length,coordinate+2),min(length,coordinate+7))
        strip = pixels[start:end,outer_start:outer_end] if axis == 1 else pixels[outer_start:outer_end,start:end].transpose(1,0,2)
        textures.append(float(np.mean(np.abs(np.diff(strip,axis=0)))) if strip.size else 100)
    return scores,textures


def _peaks(values, limit=24):
    order = np.argsort(values)[::-1]
    result = []
    for coordinate in order:
        coordinate = int(coordinate)
        if values[coordinate] < .08:
            break
        if all(abs(coordinate - previous) > 5 for previous in result):
            result.append(coordinate)
            if len(result) == limit:
                break
    return sorted(result)


def photo_rectangles(image, min_area=.10):
    """Find four long seams without searching for a person-sized crop.

    Boundary evidence is measured along each complete rectangle side. Ordinary
    object edges and curved masks generally cannot supply all four seams.
    """
    pixels = np.asarray(image, dtype=np.float32)
    h, w = pixels.shape[:2]
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    dx[:, 1:] = np.mean(np.abs(np.diff(pixels, axis=1)), axis=2)
    dy[1:, :] = np.mean(np.abs(np.diff(pixels, axis=0)), axis=2)
    # Allow the seam to vary by one pixel after preview resizing.
    vertical = np.maximum.reduce([dx, np.roll(dx, 1, axis=1), np.roll(dx, -1, axis=1)]) > 10
    horizontal = np.maximum.reduce([dy, np.roll(dy, 1, axis=0), np.roll(dy, -1, axis=0)]) > 10
    xs = sorted(set([0, w] + _peaks(vertical.mean(axis=0)*(1+dx.mean(axis=0)/15))))
    ys = sorted(set([0, h] + _peaks(horizontal.mean(axis=1)*(1+dy.mean(axis=1)/15))))
    vp = np.pad(vertical.cumsum(axis=0), ((1, 0), (0, 0)))
    hp = np.pad(horizontal.cumsum(axis=1), ((0, 0), (1, 0)))
    candidates = []
    for xi, x1 in enumerate(xs):
        for x2 in xs[xi+1:]:
            bw = x2-x1
            if bw < w*.25:
                continue
            for yi, y1 in enumerate(ys):
                for y2 in ys[yi+1:]:
                    bh = y2-y1
                    area = bw*bh/(w*h)
                    if bh < h*.18 or not min_area <= area <= .90 or not .30 <= bw/bh <= 3:
                        continue
                    supports = []
                    if x1 > 0:
                        supports.append(float(vp[y2,x1]-vp[y1,x1])/bh)
                    if x2 < w:
                        supports.append(float(vp[y2,x2]-vp[y1,x2])/bh)
                    if y1 > 0:
                        supports.append(float(hp[y1,x2]-hp[y1,x1])/bw)
                    if y2 < h:
                        supports.append(float(hp[y2,x2]-hp[y2,x1])/bw)
                    if len(supports) < 2 or min(supports) < .65 or np.mean(supports) < .78:
                        continue
                    candidates.append({"bbox":[x1,y1,x2,y2],"support":float(np.mean(supports)),"area":area,"edge_count":len(supports)})
                    strips = []
                    if x1 > 2:
                        strips.append(pixels[y1:y2,max(0,x1-5):x1])
                    if x2 < w-2:
                        strips.append(pixels[y1:y2,x2:min(w,x2+5)])
                    if y1 > 2:
                        strips.append(pixels[max(0,y1-5):y1,x1:x2].transpose(1,0,2))
                    if y2 < h-2:
                        strips.append(pixels[y2:min(h,y2+5),x1:x2].transpose(1,0,2))
                    textures = [float(np.mean(np.abs(np.diff(strip,axis=0)))) for strip in strips if len(strip)>1]
                    quiet = sum(t < 4 for t in textures)
                    candidates[-1]["quiet_edges"] = quiet
                    candidates[-1]["outside_texture"] = float(np.mean(textures)) if textures else 100
    candidates = [c for c in candidates if c["quiet_edges"] >= max(2,c["edge_count"]-1)]
    candidates.sort(key=lambda c: (round(c["support"],1), c["area"]), reverse=True)
    return candidates[:12]


def refine_photo_region(image, existing_bbox):
    """Tighten a detected canvas only when long seams corroborate its bounds.

    Never manufactures a person crop in a previously borderless/curved image.
    The original proposal remains available to review alongside this result.
    """
    if not existing_bbox:
        return None
    preview = image.copy()
    preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
    w, h = image.size
    sx, sy = preview.width/w, preview.height/h
    base = [round(v*(sx if i%2 == 0 else sy)) for i,v in enumerate(existing_bbox)]
    if not (0 <= base[0] < base[2] <= preview.width and
            0 <= base[1] < base[3] <= preview.height):
        return None
    supports,textures = boundary_evidence(preview,base)
    complete_rectangle = len(supports) == 4 and sum(value >= .80 for value in supports) >= 3
    complete_side_canvas = bool(supports) and min(supports) >= .80 and base[1]==0 and base[3]==preview.height
    if supports and max(textures) < 4 and (complete_rectangle or complete_side_canvas):
        return None
    base_area = (base[2]-base[0])*(base[3]-base[1])
    options = []
    for candidate in photo_rectangles(preview):
        box = candidate["bbox"]
        area = (box[2]-box[0])*(box[3]-box[1])
        intersection = max(0,min(box[2],base[2])-max(box[0],base[0]))*max(0,min(box[3],base[3])-max(box[1],base[1]))
        shared = sum(abs(a-b) <= max(3,(preview.width if i%2 == 0 else preview.height)*.025) for i,(a,b) in enumerate(zip(box,base)))
        if candidate["support"] < .90 or intersection/area < .98 or shared < 2 or not .35 <= area/base_area < .97:
            continue
        options.append(candidate)
    if not options:
        return None
    options.sort(key=lambda c:(round(c["support"],2)+.025*c["edge_count"], c["area"]), reverse=True)
    result = dict(options[0])
    result["bbox"] = [round(v/(sx if i%2 == 0 else sy)) for i,v in enumerate(result["bbox"])]
    return result


def expand_photo_region(image, existing_bbox):
    """Recover photo area where a weak internal edge hid a stronger outer seam.

    Only expands existing proposals, never shrinks them. Each replacement edge
    needs broad seam coverage and a quiet exterior ribbon. Ambiguous photos
    still require a human decision.
    """
    if not existing_bbox:
        return None
    preview = image.copy()
    preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
    sx, sy = preview.width / image.width, preview.height / image.height
    box = [round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(existing_bbox)]
    pixels = np.asarray(preview, dtype=np.float32)
    h, w = pixels.shape[:2]
    if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
        return None
    original = box.copy()
    for _ in range(2):
        for side in range(4):
            vertical = side % 2 == 0
            limit = w if vertical else h
            coordinate = box[side]
            # Treat both orientations as rows perpendicular to the boundary.
            region = (pixels[box[1]:box[3]].transpose(1, 0, 2) if vertical
                      else pixels[:, box[0]:box[2]])
            difference = np.mean(np.abs(np.diff(region, axis=0)), axis=2)
            gradient = np.zeros((limit, region.shape[1]), dtype=np.float32)
            gradient[1:] = difference
            band = np.maximum.reduce([gradient, np.roll(gradient, 1, axis=0),
                                      np.roll(gradient, -1, axis=0)])
            coverage = (band > 4).mean(axis=1)
            if coordinate < 3 or coordinate >= limit - 3 or coverage[coordinate] >= .70:
                continue
            minimum_move = max(6, round(limit * .015))
            maximum_move = round(limit * .40)
            positions = (range(max(3, coordinate - maximum_move), coordinate - minimum_move)
                         if side < 2 else
                         range(coordinate + minimum_move, min(limit - 3, coordinate + maximum_move)))
            options = []
            for candidate in positions:
                if coverage[candidate] < .65 or coverage[candidate] < coverage[coordinate] + .20:
                    continue
                exterior = (region[candidate-7:candidate-2] if side < 2
                            else region[candidate+2:min(limit, candidate+7)])
                if len(exterior) < 2:
                    continue
                texture = float(np.mean(np.abs(np.diff(exterior, axis=1))))
                if texture > 2:
                    continue
                score = coverage[candidate] * (1 + float(band[candidate].mean()) / 15)
                options.append((score, candidate))
            if options:
                box[side] = max(options)[1]
    if box == original:
        return None
    return {"bbox": [round(v / (sx if i % 2 == 0 else sy)) for i, v in enumerate(box)],
            "area": (box[2]-box[0]) * (box[3]-box[1]) / (w*h)}
