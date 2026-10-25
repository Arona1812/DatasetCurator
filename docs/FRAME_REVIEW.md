# Reviewing screenshot frames

The **Frames** module works after Preflight. It detects borders and proposes a photo region without changing the original file. You can accept a crop, keep the original, or exclude the image from the dataset.

## Everyday workflow

1. Run **Preflight** on the source folder.
2. Open **Frames**. Review the photo-region suggestion beside the original.
3. For straightforward uniform borders or top/bottom bars, enable **Automatic local crop decisions → Apply high confidence** if you want the app to accept those cases. Other layouts stay available for review.
4. For a difficult image, use the optional AI finder or set the crop manually. Check that the person remains fully inside the selected region.
5. If several images share a layout, use **Similar images** to preview and confirm the matching crops together. Deselect any image that needs a different boundary.

**Skip frames · go straight to audit** turns the frame module off for this dataset when no crops are needed. The audit then uses the original images and never pauses for frame proposals. Saved frame decisions stay on disk; **Analyze frames locally** turns the module on again.

Round frames and uncertain photo boundaries are kept for review instead of receiving an aggressive automatic crop. A confirmed manual decision takes precedence over automatic suggestions.

**Exclude image from dataset** is reversible. The excluded view lets you restore the image later. After changing crop or exclusion decisions, run the normal curator again before exporting; the app blocks a continuation based on stale frame audits.

The optional AI photo-region finder sends the selected image to OpenAI and incurs API costs. Its proposal needs confirmation. Local frame suggestions and manual crops do not use the API.

## Reference benchmark for development

`frame_benchmark.py` can make a separate local review page for manually confirmed photo boundaries. Use an output folder outside the source image folder:

```bash
python frame_benchmark.py Datasets/frames Datasets/frame_finder_review/benchmark
```

Open the generated `review.html`, confirm or correct boundaries, then save the JSON references. These references are manual development data, not automatic truth. Optional HEIC/HEIF inputs need `requirements-benchmark.txt`; the normal curator handles JPG, PNG, WebP, and BMP.

The scoring and comparison scripts are `frame_benchmark_score.py` and `frame_benchmark_compare.py`. Their measurements and current limitations are documented in [PROJECT_REVIEW.md](../PROJECT_REVIEW.md).
