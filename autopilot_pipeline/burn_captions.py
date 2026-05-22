import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')
_fb = os.getenv('FFMPEG_BIN','')
if _fb:
    os.environ['PATH'] = _fb + os.pathsep + os.environ.get('PATH','')

from pathlib import Path
from tools.ffmpeg_tools import measure_audio_duration
from tools.caption_tools import build_word_timings, generate_ass_file
import subprocess

video_id  = '40539d34-f145-4e63-8ee3-550323a5c9c0'
video_in  = 'outputs/video/' + video_id + '.mp4'
video_out = 'outputs/video/' + video_id + '_captioned.mp4'
ass_path  = 'outputs/video/' + video_id + '_captions.ass'

narrations = [
    'Are you making these costly financial mistakes in your 20s? Many young adults are unknowingly sabotaging their financial future.',
    'Carrying high-interest credit card debt is like having a money leak. Pay off balances monthly to avoid paying hundreds in interest.',
    'Your credit score determines your financial opportunities. Start building good credit habits now — pay on time, keep utilization low.',
    'Skipping retirement savings in your 20s costs you decades of compound growth. Even small contributions now grow massively by retirement.',
    'Using credit cards for lifestyle inflation you cannot afford is a trap. Live within your means and only charge what you can pay off.',
    'Not having a budget means money disappears without purpose. Track your income and expenses — every dollar needs a job.',
    'Investing feels scary but time is your biggest asset. Start with index funds, automate contributions, and let compounding work for you.',
    'Ignoring taxes is expensive. Learn basic tax planning — max your 401k, use an HSA, and understand deductions before tax season.',
    'Your financial foundation is built in your 20s. Every good habit you build now multiplies into wealth by your 30s and beyond.',
    'Start today. Open a savings account, pay down debt, and invest even just twenty dollars a month. Your future self will thank you.',
]

audio_dir = Path('outputs/audio')
scenes = []
for i, narr in enumerate(narrations, 1):
    af = audio_dir / (video_id + '_scene_' + str(i).zfill(3) + '.wav')
    dur = measure_audio_duration(str(af)) if af.exists() else 25.0
    scenes.append({'scene_id': i, 'narration': narr, 'duration_hint_s': dur})
    print('Scene', i, ':', round(dur, 1), 's')

timings = build_word_timings(scenes)
print('Total word events:', len(timings))

generate_ass_file(timings, ass_path, niche='personal_finance')
print('ASS written:', ass_path)

# Escape path for FFmpeg ass filter on Windows
ass_escaped = ass_path.replace('\\', '/').replace(':', '\\:')
vf_filter = "ass='" + ass_escaped + "'"
print('VF filter:', vf_filter)

result = subprocess.run(
    ['ffmpeg', '-y', '-i', video_in, '-vf', vf_filter,
     '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-c:a', 'copy', video_out],
    capture_output=True, text=True, timeout=1800
)

if result.returncode == 0:
    size = Path(video_out).stat().st_size / 1024 / 1024
    print('SUCCESS! Captioned video:', video_out, '(' + str(round(size,1)) + ' MB)')
else:
    print('FAILED stderr (last 800 chars):')
    print(result.stderr[-800:])
