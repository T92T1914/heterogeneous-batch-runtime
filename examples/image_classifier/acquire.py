"""Acquire only pinned public artifacts into an explicitly selected directory."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.request
import os
import tempfile

def publish(target, data, expected_hash):
    """Publish complete verified bytes atomically without replacing any file."""
    target=Path(target)
    if sha256(data).hexdigest()!=expected_hash:
        raise ValueError('unverified artifact bytes')
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.acquire-',suffix='.tmp',delete=False) as output:
            temporary=Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            # A hard link is an atomic exclusive publication on supported local
            # filesystems. Existing content is never overwritten.
            os.link(temporary,target)
        except FileExistsError:
            if sha256(target.read_bytes()).hexdigest()!=expected_hash:
                raise ValueError('concurrently created artifact differs; preserved')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

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
        # Name the pinned input without logging a redirect URL or response body.
        print(f"Acquiring pinned artifact {record['name']} ({record['bytes']} expected bytes)", flush=True)
        with urllib.request.urlopen(request,timeout=30) as response:
            data=response.read(record['bytes']+1)
        observed_hash=sha256(data).hexdigest()
        if len(data)!=record['bytes'] or observed_hash!=record['sha256']:
            raise ValueError(
                f"acquired artifact {record['name']} failed size or identity validation: "
                f"expected {record['bytes']} bytes and sha256 {record['sha256']}; "
                f"received {len(data)} bytes and sha256 {observed_hash}; "
                'no artifact published')
        publish(target,data,record['sha256'])
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('destination')
    args=parser.parse_args()
    acquire(args.destination)
    print('Pinned model and API headers verified. No provider was initialized.')
