"""Acquire only pinned public artifacts into an explicitly selected directory."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.error
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

def fetch(record):
    """Read at most one byte past the pinned size from the first source that answers.

    A source that cannot be reached, or whose response breaks off with a
    network error before the read completes, passes the request to the next
    one. Its partial bytes are dropped. Bytes that do arrive in full are
    returned as they are, so an identity failure stops acquisition and is
    never retried from another source.
    """
    sources=[record['url'],*record.get('fallback_urls',[])]
    failures=[]
    for position,url in enumerate(sources,1):
        request=urllib.request.Request(url,headers={'User-Agent':'hbr-image-classifier/0.1'})
        try:
            with urllib.request.urlopen(request,timeout=30) as response:
                return response.read(record['bytes']+1)
        except urllib.error.HTTPError as error:
            failures.append(f'HTTP {error.code}')
        except OSError as error:
            failures.append(type(error).__name__)
        if position<len(sources):
            print(f"Source {position} of {len(sources)} for {record['name']} failed "
                  f"({failures[-1]}). Trying source {position+1}.", flush=True)
    raise ConnectionError(
        f"no source answered for {record['name']}: {', '.join(failures)}; no artifact published")

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
        # Name the pinned input without logging a redirect URL or response body.
        print(f"Acquiring pinned artifact {record['name']} ({record['bytes']} expected bytes)", flush=True)
        data=fetch(record)
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
