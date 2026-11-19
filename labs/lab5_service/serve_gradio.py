"""Gradio demo around `TTSPipeline` — extra task 3, browser variant.

    uv pip install gradio
    python serve_gradio.py          # 127.0.0.1:7860, plus a public link if TTS_SHARE=1

The pipeline is built once at import: loading FastSpeech2, the pause predictor and
HiFi-GAN per request would dominate the latency.
"""
import os
import tempfile
from pathlib import Path

import gradio as gr
import soundfile as sf

from tts_pipeline import SAMPLE_RATE, TTSPipeline

BASE = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ.get('TTS_RUN_DIR', BASE.parent / 'lab4_acoustic' / 'runs' / 'baseline_gt'))
DEVICE = os.environ.get('TTS_DEVICE', 'cuda')

pipeline = TTSPipeline(RUN_DIR, device=DEVICE)
pipeline('Разминка перед первым запросом.', markup=False)

EXAMPLES = [
    ['Здравствуйте, ваш заказ готов.', False],
    ['Казнить <pause type="normal"> нельзя, <pause type="none"> помиловать.', True],
    ['Мы всем отрядом съели все, что было во всем лагере.', False],
]


def synthesize(text: str, markup: bool) -> tuple[str, str]:
    """Text and the markup flag to a wav path and a one-line timing report."""
    if not text or not text.strip():
        raise gr.Error('Пустой текст: введите фразу')
    wav, timing = pipeline(text, markup=bool(markup))
    if not wav.size:
        raise gr.Error('После нормализации не осталось текста для синтеза')

    path = Path(tempfile.mkdtemp()) / 'synthesis.wav'
    sf.write(path, wav, SAMPLE_RATE, subtype='PCM_16')
    report = (f'RTF {timing.rtf:.3f} · синтез {timing.total * 1000:.0f} мс · '
              f'звук {timing.audio_seconds:.2f} с')
    return str(path), report


demo = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(label='Текст', lines=4, placeholder='Введите фразу на русском'),
        gr.Checkbox(label='Разметка пауз (<pause type="normal">, <pause type="none">)',
                    value=False),
    ],
    outputs=[
        gr.Audio(label='Синтез', type='filepath'),
        gr.Textbox(label='Замеры', interactive=False),
    ],
    title='Синтез речи: RUSLAN TTS',
    description=('FastSpeech2 (ваша модель из работы №4) + HiFi-GAN. '
                 'Теги пауз работают только при включённой галочке.'),
    examples=EXAMPLES,
    cache_examples=False,
)


if __name__ == '__main__':
    demo.queue().launch(
        server_name=os.environ.get('TTS_HOST', '127.0.0.1'),
        server_port=int(os.environ.get('TTS_PORT', 7860)),
        share=os.environ.get('TTS_SHARE', '1') == '1',
    )
