"""Acquire only pinned public artifacts into an explicitly selected directory."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

def acquire(destination):
    destination=Path(destination)
    manifest=json.loads(Path(__file__).with_name('artifacts.json').read_text())
    for record in [manifest['model'],*manifest['headers']]:
        parent=destination if record['name'].endswith('.onnx') else destination/'sdk'/'include'
        parent.mkdir(parents=True,exist_ok=True)
        target=parent/record['name']
        if target.exists():
            data=target.read_bytes()
            if sha256(data).hexdigest()!=record['sha256']:
                raise ValueError('existing artifact differs; preserved without overwrite')
            continue
        request=urllib.request.Request(record['url'],headers={'User-Agent':'hbr-image-classifier/0.1'})
        with urllib.request.urlopen(request,timeout=30) as response:
            data=response.read(record['bytes']+1)
        if len(data)!=record['bytes'] or sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('acquired artifact failed size or identity validation')
        # Exclusive create protects a concurrently acquired or changed artifact.
        with target.open('xb') as output:output.write(data)
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('destination')
    args=parser.parse_args()
    acquire(args.destination)
    print('Pinned model and API headers verified. No provider was initialized.')
