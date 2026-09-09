"""Download the pinned PlantWild v2 archive for local research, verifying SHA256."""
import hashlib
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance/field_improvement'
URL = 'https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/527a72eb8f00c95e41698bb09d982c7d4625dce3/plantwild_v2.zip'
SHA256 = '5c3fd0c6ecf9a346c9d2dd03ff47769c71e581eb1170a68477d718ab43cfee5b'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'plantwild_v2.zip'
    if target.exists():
        with target.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == SHA256
        print('Verified cached PlantWild v2 archive.', flush=True)
        return
    partial = target.with_suffix('.zip.part')
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    with requests.get(URL + f'?resume={offset}', headers=headers, stream=True, timeout=(30, 60)) as response:
        response.raise_for_status()
        if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
            raise RuntimeError('Server did not honor resume range; partial download preserved.')
        digest = hashlib.sha256()
        if offset:
            with partial.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        count, last = offset, time.monotonic()
        with partial.open('ab' if offset else 'wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if time.monotonic() - last > 15:
                    print(f'Downloaded {count / 1024**2:.0f} MiB of 1564 MiB', flush=True)
                    last = time.monotonic()
        if digest.hexdigest() != SHA256:
            raise RuntimeError('Archive checksum mismatch; candidate data not accepted.')
    partial.replace(target)
    print('PlantWild v2 downloaded and checksum verified.', flush=True)


if __name__ == '__main__':
    main()
