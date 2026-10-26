"""Create a local visual comparison of detector versions on fixed references."""
from __future__ import annotations
import os as _os
import sys as _sys

# The detector modules live in ../app.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "app"))
import argparse
import base64
from html import escape
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw
from frame_cleanup import load_bounded_rgb


def crop_preview(source, box):
    image = load_bounded_rgb(source["source_path"],900)
    if box:
        w,h = source["size"]
        scaled = [round(v*(image.width/w if i%2==0 else image.height/h)) for i,v in enumerate(box)]
        image = image.crop(scaled)
    image.thumbnail((280,380))
    return image


def comparison(before, after, manifest, output):
    old = {r["id"]:r for r in before["results"]}
    sources = {r["id"]:r for r in manifest["records"]}
    changed = [r for r in after["results"] if r["id"] in old and old[r["id"]]["local_bbox"] != r["local_bbox"]]
    changed.sort(key=lambda r:r.get("iou",0)-old[r["id"]].get("iou",0),reverse=True)
    regressions = [r["filename"] for r in changed if old[r["id"]].get("iou",0)-r.get("iou",0)>.01]
    lost = [r["filename"] for r in after["results"] if r["id"] in old and old[r["id"]]["outcome"]=="near_reference" and r["outcome"]!="near_reference"]
    output.mkdir(parents=True,exist_ok=True)
    summary = {"confirmed":after["confirmed"],"changed":len(changed),"before_near":before["outcomes"].get("near_reference",0),
               "after_near":after["outcomes"].get("near_reference",0),"before_extra":before["substantial_extra_area"],
               "after_extra":after["substantial_extra_area"],"before_missing":before["substantial_missing_photo"],
               "after_missing":after["substantial_missing_photo"],"worse_overlap":regressions,"lost_previous_near":lost}
    (output/"comparison.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    intro = f"{len(changed)} geänderte Vorschläge im Vergleich mit {after['confirmed']} bestätigten Referenzen. "
    intro += "Die Referenzen sind Handmarkierungen. Sie wurden bei der Entwicklung verwendet; dieser Vergleich ist kein unabhängiger Abnahmetest. Vorschläge bleiben prüfpflichtig."
    cards = []
    sheet = Image.new("RGB",(900,min(6,len(changed))*420),"#172033")
    draw = ImageDraw.Draw(sheet)
    for index,row in enumerate(changed):
        source = sources[row["id"]]
        columns = []
        for column,(label,box) in enumerate((("Bisher",old[row["id"]]["local_bbox"]),("Verbessert",row["local_bbox"]),("Deine Handmarkierung",row["reference_bbox"]))):
            preview = crop_preview(source,box)
            buffer = io.BytesIO();preview.save(buffer,format="JPEG",quality=88)
            url = "data:image/jpeg;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")
            columns.append(f'<figure><figcaption>{label}</figcaption><img src="{url}" alt="{label}"></figure>')
            if index<6:
                x,y = column*300,index*420
                sheet.paste(preview,(x+(300-preview.width)//2,y+30))
                draw.text((x+10,y+8),label,fill="white")
        if index<6:
            draw.text((10,index*420+397),row["filename"][:85],fill="white")
        cards.append(f'<section><h2>{escape(row["filename"])}</h2><div class="images">'+"".join(columns)+"</div></section>")
    if changed:
        sheet.save(output/"comparison_sheet.jpg",quality=90)
    table = ("| Messung | Bisher | Verbessert |\n| --- | ---: | ---: |\n"
             f"| Fotogrenzen nahe Referenz (0,5 % / mindestens 3 Pixel) | {summary['before_near']} | {summary['after_near']} |\n"
             f"| Mehr als 10 % zusätzliche Außenfläche | {summary['before_extra']} | {summary['after_extra']} |\n"
             f"| Mehr als 5 % fehlende Fotofläche | {summary['before_missing']} | {summary['after_missing']} |\n")
    (output/"COMPARISON.md").write_text("# Lokaler Rahmenfinder: Vorher und nachher\n\n"+intro+"\n\n"+table+
        f"\nZuvor nahe Vorschläge verschlechtert: {len(lost)}. Überlappung um mehr als einen Prozentpunkt verschlechtert: {len(regressions)}. "
        f"Bestätigte Originale ohne Zuschnittsvorschlag: {after['outcomes'].get('correct_keep', 0)}. "
        f"Zuschneidbare Referenzen ohne Vorschlag: {after['outcomes'].get('missing_proposal', 0)}. "
        "Weitere komplexe Story-Layouts brauchen Nacharbeit.\n\n"
        "[Bilder vergleichen](comparison.html) · [Einzelauswertung](SCORE_REPORT.md)\n",encoding="utf-8")
    html = '<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rahmenfinder: Vorher und nachher</title><style>body{background:#111827;color:#edf2fa;font:16px system-ui;margin:24px auto;max-width:1100px;padding:16px}p{line-height:1.6;color:#bbc9df}section{background:#1b283e;padding:20px;border-radius:12px;margin:20px 0}h2{font-size:18px;overflow-wrap:anywhere}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0;text-align:center}figcaption{margin-bottom:12px}img{max-width:100%;max-height:380px;object-fit:contain}@media(max-width:600px){.images{grid-template-columns:1fr}}</style><h1>Rahmenfinder: Vorher und nachher</h1><p>'+escape(intro)+"</p>"+"".join(cards)+"</html>"
    (output/"comparison.html").write_text(html,encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("before","after","manifest","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    read=lambda path:json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(comparison(read(args.before),read(args.after),read(args.manifest),args.output),ensure_ascii=False))
