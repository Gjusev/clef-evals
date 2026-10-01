# clef-evals: composition brief

Deliverable: a 14 second showcase video for the clef-evals README and social
shares, plus a 1920x1080 cover still. Both rendered from `render_video.py`
(PIL frames, ffmpeg h264 encode). No stock footage, no gradients, no emoji.

## Style rules (minimalist-ui)

- Canvas: warm bone #F7F6F3. Text: charcoal #2F3437. Muted text: #787774.
- One accent: muted terracotta #D97A3D. Used only for motion and the gate dot.
- Cards: white, 1px #EAEAEA border, 8px radius, generous padding.
- Type: Helvetica Neue for display, JetBrains Mono / Consolas for code and numbers.
- Motion: transform/opacity only, eased, never more than two moving things per scene.

## Scenes (14s @ 30fps)

| Time | Scene | Beat |
|---|---|---|
| 0.0-2.4 | Title | "clef-evals" fades up, tagline settles, accent dot breathes |
| 2.4-6.4 | Pipeline | Three cards cascade in, accent dots travel the wires into ClefClient |
| 6.4-10.0 | Benchmarks | Published Decision Index bars animate: Clef vs Laya, three benchmarks |
| 10.0-12.6 | Gate | Terminal types the CLI command, result line lands "gate: PASSED" |
| 12.6-14.0 | Close | "pip install clef-evals" and the repo URL, accent dot holds |

## Numbers used on screen

Published reference only (huggingface.co/Cloudflare/clef, Decision Index 0.2.1):
BFCL 98.5 vs 38.1, BANKING77 94.2 vs 14.3, CLINC150 97.4 vs 3.2. Sourced on
screen with a small "source: Cloudflare Decision Index 0.2.1" line. No invented
measurements.

## Render

    python brag-output/render_video.py   # writes brag.mp4 and brag.jpg
