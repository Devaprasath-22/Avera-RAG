import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import resolve_language, SUPPORTED_LANGUAGES

test_inputs = [
    "Voice:en", "voice:ta", "Voice:hi", "Voice:te", "Voice:ka",
    "voice:kn", "1", "2", "3", "4", "5",
    "en", "ta", "hi", "te", "ka", "kn",
    "tamil", "hindi", "telugu", "kannada", "english"
]

print(f"{'Input':15} | {'Code':5} | {'Voice Tag':10} | {'Native'}")
print("-" * 50)
for inp in test_inputs:
    res = resolve_language(inp)
    print(f"{inp:15} | {res['code']:5} | {res['voice_tag']:10} | {res['native']}")

print("\n--- Menu Simulation ---")
print("choose Avera Available lang: Voice:en, voice:ta, Voice:hi, Voice:te, Voice:ka\n")
for k, v in SUPPORTED_LANGUAGES.items():
    print(f"  [{k}] {v['voice_tag']} — {v['name']} ({v['native']})")

for sample in ["Voice:ka", "voice:ta", "3", "te", "Voice:en"]:
    resolved = resolve_language(sample)
    print(f"User enters '{sample}' -> Active: {resolved['voice_tag']} ({resolved['native']})")
