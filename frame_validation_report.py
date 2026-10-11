"""Render frozen validation results without changing proposals or references."""
import argparse
import base64
from html import escape
import io
import json
from pathlib import Path

from frame_benchmark_compare import crop_preview


def render(manifest, scores, output):
    sources = {r['id']: r for r in manifest['records']}
    automatic_errors = {'needs_correction', 'unnecessary_automatic_crop', 'special_case_automatic_crop', 'invalid_automatic_crop'}
    labels = {'near_reference': 'Nahe deiner Referenz', 'needs_correction': 'Grenzen weichen ab',
              'missing_proposal': 'Vorschlag fehlt', 'correct_keep': 'Original korrekt beibehalten',
              'unnecessary_proposal': 'Unnötiger Vorschlag', 'special_case_without_proposal': 'Sonderfall ohne Vorschlag',
              'special_case_with_proposal': 'Sonderfall mit Vorschlag'}
    rows = sorted(scores['results'], key=lambda r: (
        r.get('automatic_outcome') not in automatic_errors,
        -r.get('missing_photo_fraction', 0), -r.get('extra_area_fraction', 0)))
    cards = []
    for row in rows:
        source = sources[row['id']]
        bad_auto = row.get('automatic_outcome') in automatic_errors
        kind = 'automatic' if bad_auto else 'keep' if row['reference_decision'] == 'keep_original' else 'error' if row['outcome'] != 'near_reference' else 'correct'
        columns = []
        for label, box in (('Original', None), ('Lokaler Vorschlag', row['local_bbox']), ('Deine Referenz', row['reference_bbox'])):
            preview = crop_preview(source, box)
            buffer = io.BytesIO(); preview.save(buffer, format='JPEG', quality=88)
            url = 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
            columns.append(f'<figure><figcaption>{label}</figcaption><img src="{url}" alt="{label}"></figure>')
        automation = 'Automatik benötigt Korrektur' if bad_auto else 'Automatik aktiv' if row.get('automatic_bbox') else 'Kein automatischer Zuschnitt'
        detail = f"Fotofläche fehlt: {row.get('missing_photo_fraction', 0):.1%} · zusätzliche Außenfläche: {row.get('extra_area_fraction', 0):.1%}" if 'iou' in row else ''
        cards.append(f'<section data-kind="{kind}"><h2>{escape(row["filename"])}</h2><p>{labels[row["outcome"]]} · {automation}</p><p>{detail}</p><div class="images">'+''.join(columns)+'</div></section>')
    auto = scores.get('automatic_outcomes', {})
    summary = (f'{scores["confirmed"]} bestätigte Bilder. {auto.get("near_reference", 0)} automatische Zuschnitte nahe deiner Referenz; '
               f'{sum(auto.get(k, 0) for k in automatic_errors)} automatische Zuschnitte benötigen Nacharbeit. '
               f'{scores["outcomes"].get("correct_keep", 0)} Originale korrekt ohne Vorschlag; '
               f'{scores["outcomes"].get("unnecessary_proposal", 0)} unnötige Vorschläge.')
    html = '''<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rahmenprüfung: Auswertung</title><style>body{background:#111827;color:#eef2ff;font:16px system-ui;max-width:1100px;margin:auto;padding:24px;line-height:1.5}h1{font-size:28px}h2{font-size:18px;overflow-wrap:anywhere}section{background:#1c273a;padding:18px;margin:24px 0;border-radius:12px}p{color:#c8d2e4}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0;text-align:center}figcaption{margin-bottom:10px}img{max-width:100%;max-height:380px;object-fit:contain}select{font:inherit;padding:10px;border-radius:8px;background:#202d43;color:white}a{color:#93c5fd}[hidden]{display:none!important}@media(max-width:650px){.images{grid-template-columns:1fr}}</style>
<h1>Deine Rahmenreferenzen: Auswertung</h1>'''+f'<p>{summary}</p>'+'''
<p>Originale und eingefrorene Vorschläge bleiben unverändert. Kleine Abweichungen können durch Handmarkierungen entstehen: „nahe“ bedeutet hier maximal 3 Pixel oder 0,5 % je Bildkante. Zusätzliche Außenfläche und abgeschnittene Fotofläche sind getrennte Fehler. Dies sind lokale Ergebnisse ohne neue KI-Anfragen.</p>
<label for="filter">Anzeige </label><select id="filter"><option value="automatic">Automatik: Fälle mit Nacharbeit</option><option value="errors">Alle Fälle mit Nacharbeit</option><option value="keep">Bestätigte Originale</option><option value="all">Alle 48 Referenzen</option></select>
<p><a href="scored/SCORE_REPORT.md">Vollständige Zahlen</a> · <a href="review.html">Zur Referenzprüfung</a></p>'''+''.join(cards)+'''
<script>const filter=document.getElementById('filter');function update(){for(const card of document.querySelectorAll('section')){const kind=card.dataset.kind;card.hidden=filter.value!=='all'&&!(filter.value==='errors'?['automatic','error'].includes(kind):kind===filter.value)}}filter.onchange=update;update();</script></html>'''
    output.write_text(html, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'scores', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    render(json.loads(args.manifest.read_text(encoding='utf-8')),
           json.loads(args.scores.read_text(encoding='utf-8')), args.output)
