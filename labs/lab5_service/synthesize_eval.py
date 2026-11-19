"""Synthesize the lab 5 evaluation phrases and report per-stage time and RTF.

    uv run python synthesize_eval.py --run ../lab4_acoustic/runs/baseline_gt
    uv run python synthesize_eval.py --run ../lab4_acoustic/runs/baseline_gt \
        --markup --texts data/evaluation_text.csv data/evaluation_text_pause.csv

Writes one wav per phrase into `audio/` and `audio/timings.csv` (pipe separated) with the
median over `--repeats` runs. The first call after model loading is a warm-up and is not
measured: it pays for CUDA kernels and lazy imports, not for synthesis.
"""
import argparse
import csv
import statistics
import sys
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from tts_pipeline import TTSPipeline  # noqa: E402

STAGES = ('normalize', 'tokenize', 'pauses', 'g2p', 'acoustic', 'vocoder')
COLUMNS = ('id', 'chars', 'audio_s', 'total_ms', 'rtf', *(f'{s}_ms' for s in STAGES))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True,
                   help='lab 4 run directory with config.json and best_model_*.pth')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, default=BASE / 'audio')
    p.add_argument('--texts', type=Path, nargs='+',
                   default=[BASE / 'data/evaluation_text.csv'])
    p.add_argument('--markup', action='store_true',
                   help='parse <pause type=...> tags before normalizing (extra task 1)')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def read_texts(paths: list[Path], limit: int | None) -> list[tuple[str, str]]:
    """Read `id|text` files, skipping the header."""
    rows: list[tuple[str, str]] = []
    for path in paths:
        with path.open(encoding='utf-8') as fh:
            for line in fh:
                cols = line.rstrip('\n').split('|', 1)
                if len(cols) == 2 and cols[0] != 'id':
                    rows.append((cols[0], cols[1]))
    return rows[:limit] if limit else rows


def main() -> None:
    args = parse_args()
    rows = read_texts(args.texts, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f'модель: {args.run}, устройство: {args.device}, фраз: {len(rows)}, '
          f'повторов: {args.repeats}, разметка пауз: {args.markup}')
    pipeline = TTSPipeline(args.run, device=args.device)
    pipeline('Разминка перед замерами.', markup=False)   # прогрев, не измеряется

    records = []
    failures = 0
    for utt_id, text in rows:
        wavs, timings = [], []
        try:
            for _ in range(args.repeats):
                wav, timing = pipeline(text, markup=args.markup)
                wavs.append(wav)
                timings.append(timing)
        except Exception:                                  # noqa: BLE001
            failures += 1
            print(f'{utt_id}: ОШИБКА')
            traceback.print_exc()
            continue

        if not wavs[0].size:
            failures += 1
            print(f'{utt_id}: пустой звук')
            continue

        pipeline.save(wavs[0], args.out / f'{utt_id}.wav')
        record = {
            'id': utt_id,
            'chars': len(text),
            'audio_s': statistics.median(t.audio_seconds for t in timings),
            'total_ms': statistics.median(t.total for t in timings) * 1000,
            'rtf': statistics.median(t.rtf for t in timings),
        }
        for stage in STAGES:
            record[f'{stage}_ms'] = statistics.median(t.stages.get(stage, 0.0) for t in timings) * 1000
        records.append(record)
        print(f'{utt_id}  {record["chars"]:5d} симв.  {record["audio_s"]:6.2f} с  '
              f'{record["total_ms"]:8.0f} мс  RTF {record["rtf"]:5.3f}')

    if not records:
        raise SystemExit('ни одна фраза не синтезирована')

    with (args.out / 'timings.csv').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, COLUMNS, delimiter='|', quoting=csv.QUOTE_NONE)
        writer.writeheader()
        writer.writerows({k: (f'{v:.3f}' if isinstance(v, float) else v) for k, v in r.items()}
                         for r in records)

    # длительность ступеней по всей выборке: доля от суммы, чтобы видеть узкое место
    total_per_stage = {s: sum(r[f'{s}_ms'] for r in records) for s in STAGES}
    all_ms = sum(total_per_stage.values())
    print(f'\nсуммарно {all_ms / 1000:.1f} с на {len(records)} фраз, '
          f'медианный RTF {statistics.median(r["rtf"] for r in records):.3f}')
    print(f'{"этап":<12}{"мс":>10}{"доля":>8}')
    for stage in STAGES:
        print(f'{stage:<12}{total_per_stage[stage]:>10.0f}{total_per_stage[stage] / all_ms:>8.1%}')
    print(f'\nзаписано {len(records)} wav и {args.out / "timings.csv"}')
    if failures:
        raise SystemExit(f'с ошибками: {failures}')


if __name__ == '__main__':
    main()
