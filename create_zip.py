import zipfile
import os

files = [
    'project_starter.py',
    'test_results.csv',
    'munder_difflin.db',
    'quote_requests.csv',
    'quote_requests_sample.csv',
    'quotes.csv'
]

print("Creating submission.zip...")

with zipfile.ZipFile('submission.zip', 'w') as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
            print(f"Added: {f}")
        else:
            print(f"MISSING: {f}")

print("\nVerifying contents of submission.zip:")
with zipfile.ZipFile('submission.zip', 'r') as z:
    for name in z.namelist():
        print(f"  - {name}")

print("\nDone! submission.zip is ready to submit.")