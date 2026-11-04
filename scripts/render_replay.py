"""Render an explicitly labeled, subsampled animation from recorded observations."""

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'assets/replay.gif')
    args = p.parse_args()
    records = [json.loads((ROOT / f'experiments/v3/heldout-{name}/replay.json').read_text()) for name in ['ours', 'reference']]
    assert all(r['episode']['seed'] == 200000 for r in records)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=22)
    colors = {'bg': '#121823', 'grid': '#293344', 'ball': '#f6f8fb', 'paddle': '#4fc3f7', 'brick': '#f7b267', 'text': '#e5eaf1'}
    images = []
    last = max(len(r['frames']) for r in records) - 1
    for step in sorted(set(range(0, last + 1, 20)) | {last}):
        canvas = Image.new('RGB', (640, 430), colors['bg'])
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 8), 'Real replay | seed 200000 | every 20 steps', font=small, fill=colors['text'])
        for panel, (name, record) in enumerate(zip(['Our recurrent network', 'Released reference'], records)):
            f = record['frames'][min(step, len(record['frames']) - 1)]
            x0, y0, cell = 16 + panel * 320, 86, 29
            draw.text((x0, 35), name, font=font, fill=colors['text'])
            ended = step >= len(record['frames']) - 1
            status = ' | time cap' if ended and not record['episode']['terminated'] else ' | ended' if ended else ''
            draw.text((x0, 61), f"Score {f['s']}{status}", font=small, fill=colors['text'])
            for n in range(11):
                draw.line((x0 + n * cell, y0, x0 + n * cell, y0 + 10 * cell), fill=colors['grid'])
                draw.line((x0, y0 + n * cell, x0 + 10 * cell, y0 + n * cell), fill=colors['grid'])
            for pos in f['k']:
                x, y = x0 + (pos % 10) * cell, y0 + (pos // 10) * cell
                draw.rectangle((x + 3, y + 3, x + cell - 3, y + cell - 3), fill=colors['brick'])
            for pos in f['t']:
                x, y = x0 + (pos % 10 + .5) * cell, y0 + (pos // 10 + .5) * cell
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=colors['ball'])
            x, y = x0 + f['p'] * cell, y0 + 9 * cell
            draw.rectangle((x + 2, y + 16, x + cell - 2, y + 24), fill=colors['paddle'])
            for pos in f['b']:
                x, y = x0 + (pos % 10 + .5) * cell, y0 + (pos // 10 + .5) * cell
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors['ball'])
        draw.text((16, 393), f'Step {step}/{last} | Real trajectory, redrawn', font=small, fill=colors['text'])
        images.append(canvas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    durations = [60] * len(images)
    durations[0] = 800
    durations[-1] = 2200
    images[0].save(args.output, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(args.output, len(images), 'frames')


if __name__ == '__main__':
    main()
