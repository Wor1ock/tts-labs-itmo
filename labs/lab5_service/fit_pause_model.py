"""Fit the lab 2 pause predictor once and store it for the lab 5 pipeline.

`TTSPipeline` must not retrain CatBoost on every start, so the fitted models and the
`prev_word`/`next_word` vocabulary are written to `labs/lab2_pauses/data/`.
Run once after the lab 2 metadata or feature code changes::

    uv run python fit_pause_model.py
"""
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS / 'lab2_pauses'))

from features import load_features  # noqa: E402
from pause_predictor import calc_metrics  # noqa: E402
from pause_predictor_catboost import PausePredictorCatboost, predict_on  # noqa: E402


def main() -> None:
    df = load_features()
    train_df, test_df = df[df.set == 'train'], df[df.set == 'test']
    print(f'train: {len(train_df)} строк, test: {len(test_df)} строк')

    pp = PausePredictorCatboost(verbose=100).fit(train_df)
    for name, part in (('train', train_df), ('test', test_df)):
        print(f'\n-- {name} --')
        calc_metrics(predict_on(pp, part))

    pp.save()


if __name__ == '__main__':
    main()
