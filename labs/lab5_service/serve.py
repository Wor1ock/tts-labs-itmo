"""HTTP service around `TTSPipeline` — extra task 3.

    uv pip install fastapi uvicorn
    python serve.py               # 127.0.0.1:8000 by default

    curl -X POST localhost:8000/synthesize \
         -H 'Content-Type: application/json' \
         -d '{"text": "Здравствуйте, ваш заказ готов."}' -o out.wav -D -

Models are loaded once in the lifespan hook, not per request: loading FastSpeech2 and
HiFi-GAN takes seconds and would put the RTF far above 1 on every call. The response
carries the measured timings in headers, so a client gets the RTF without a second call.

Environment:
    TTS_RUN_DIR   lab 4 run directory (default: ../lab4_acoustic/runs/baseline_gt)
    TTS_DEVICE    `cuda` or `cpu`
    TTS_HOST      bind address, default 127.0.0.1 — use an SSH tunnel to reach it
    TTS_PORT      default 8000
"""
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from tts_pipeline import SAMPLE_RATE, TTSPipeline

BASE = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ.get('TTS_RUN_DIR', BASE.parent / 'lab4_acoustic' / 'runs' / 'baseline_gt'))
DEVICE = os.environ.get('TTS_DEVICE', 'cuda')

pipeline: TTSPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print(f'loading pipeline: {RUN_DIR} on {DEVICE}', flush=True)
    pipeline = TTSPipeline(RUN_DIR, device=DEVICE)
    pipeline('Разминка перед первым запросом.', markup=False)   # CUDA kernels, lazy imports
    print('pipeline ready', flush=True)
    yield
    pipeline = None


app = FastAPI(title='RUSLAN TTS', lifespan=lifespan)


class SynthRequest(BaseModel):
    """One synthesis request. `markup` enables `<pause type=...>` tags."""

    text: str
    markup: bool = False


@app.get('/')
def root() -> dict:
    return {'service': 'RUSLAN TTS', 'endpoints': ['/synthesize', '/health'],
            'request': {'text': 'str', 'markup': 'bool'}}


@app.get('/health')
def health() -> dict:
    return {'status': 'ok' if pipeline is not None else 'loading',
            'model': str(RUN_DIR), 'device': DEVICE}


@app.post('/synthesize')
def synthesize(request: SynthRequest) -> Response:
    """Text in, `audio/wav` out, timings in headers. Never raises out of the process."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail='pipeline is still loading')
    if not request.text.strip():
        raise HTTPException(status_code=400, detail='empty text')

    try:
        wav, timing = pipeline(request.text, markup=request.markup)
    except Exception as exc:                        # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f'{type(exc).__name__}: {exc}') from exc

    if not wav.size:
        raise HTTPException(status_code=422, detail='nothing left to synthesize after normalization')

    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format='WAV', subtype='PCM_16')
    return Response(
        content=buffer.getvalue(),
        media_type='audio/wav',
        headers={
            'X-RTF': f'{timing.rtf:.3f}',
            'X-Synthesis-Ms': f'{timing.total * 1000:.0f}',
            'X-Audio-Seconds': f'{timing.audio_seconds:.2f}',
        },
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host=os.environ.get('TTS_HOST', '127.0.0.1'),
                port=int(os.environ.get('TTS_PORT', 8000)))
