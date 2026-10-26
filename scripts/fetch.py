"""Downloads the files of the project and verifies their SHA-256 (standard library only).

  model    trained SegFormer-B0 (best epoch 76)          -> checkpoints/segformer_b0_france_ep76.pth
  convnext trained DINOv3 ConvNeXt-B + UPerNet (epoch 43) -> checkpoints/convnext_b_upernet_france_ep43.pth
  init     MiT-B0 encoder pretrained on LoveDA (training) -> checkpoints/mit_b0_loveda_encoder.pth
  mmcv     MMCV 2.1.0 wheel for torch 2.1 / CUDA 12.1 / Python 3.11 (Linux; --windows for the Windows build)
  dataset  the France dataset from its public Yandex Disk folder -> <out>/{City,Village,Other}/*.tif
Release, file names and checksums are in env/assets.json. The dataset needs no account or token: the public link is
resolved to direct download links by Yandex Disk's public download service, file by file, and every file is checked
against its SHA-256. Files already present with the right checksum are not downloaded again.

Every file is tried with a chain of methods: Python, curl, PowerShell (Windows) or wget (Linux), each with retries on
network errors. If no secure connection can be established (a network that substitutes certificates: corporate
proxy, antivirus), the download is repeated without certificate verification; the SHA-256 check still protects it.
Usage: python scripts/fetch.py model|convnext|init|mmcv [--windows] [--out FILE] | python scripts/fetch.py dataset --out DIR"""
import argparse
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
ASSETS = json.loads((BUNDLE / 'env' / 'assets.json').read_text(encoding='utf-8'))
WIN = os.name == 'nt'
UA = {'User-Agent': 'france-segformer-b0'}
SYSTEM_CA = ['/etc/ssl/certs/ca-certificates.crt', '/etc/pki/tls/certs/ca-bundle.crt', '/etc/ssl/ca-bundle.pem',
             '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem', '/etc/ssl/cert.pem']
CERT_WORDS = ('CERTIFICATE', 'SSL', 'TLS', 'SCHANNEL', 'CURL 60', 'CURL 35', 'CURL 77', 'TRUST', 'CERT_',
              'СЕРТИФИКАТ', 'ДОВЕРИТЕЛЬН', 'ЗАЩИЩЕННОГО КАНАЛА', 'ЗАЩИЩЁННОГО КАНАЛА')


def log(msg):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


class Missing(Exception):
    """No such file on the server or the method is unavailable: retrying this method is pointless."""


def by_python(url, part, insecure):
    if insecure:
        ctx = ssl._create_unverified_context()
    else:
        ctx = ssl.create_default_context()
        for ca in SYSTEM_CA:                     # system certificates, including those added by an administrator
            if os.path.exists(ca):
                try:
                    ctx.load_verify_locations(ca)
                except (OSError, ssl.SSLError):
                    pass
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120, context=ctx)
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500:
            raise Missing(f'HTTP {e.code}') from e
        raise
    with r, open(part, 'wb') as f:
        total, done, shown, t0 = int(r.headers.get('Content-Length') or 0), 0, -10, time.time()
        for chunk in iter(lambda: r.read(1 << 22), b''):
            f.write(chunk)
            done += len(chunk)
            pct = 100 * done // total if total else 0
            if total > 20 * 2 ** 20 and (pct >= shown + 10 or done == total):
                shown = pct
                log(f'  {done / 2 ** 20:.0f}/{total / 2 ** 20:.0f} MB ({done / 2 ** 20 / max(time.time() - t0, 1e-3):.1f} MB/s)')


def curl_exe():
    if WIN:
        exe = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'curl.exe')
        if os.path.exists(exe):
            return exe
    return shutil.which('curl')


def by_curl(url, part, insecure):
    exe = curl_exe()
    if not exe:
        raise Missing('curl not found')
    cmd = [exe, '--fail', '--location', '--silent', '--show-error', '--connect-timeout', '30', '--output', part, url]
    if insecure:
        cmd.insert(1, '--insecure')
    r = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    if r.returncode and WIN and not insecure:    # Windows: revocation servers are often closed in corporate networks
        r = subprocess.run(cmd[:1] + ['--ssl-revoke-best-effort'] + cmd[1:], capture_output=True, text=True,
                           errors='replace')
    if r.returncode:
        raise RuntimeError(f'curl {r.returncode}: {r.stderr.strip()[:600]}')


def by_powershell(url, part, insecure):
    trust = '[Net.ServicePointManager]::ServerCertificateValidationCallback={$true}; ' if insecure else ''
    ps = ("$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; "
          "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; " + trust +
          f"Invoke-WebRequest -UseBasicParsing -Uri '{url}' -OutFile '{part}' -TimeoutSec 1800")
    r = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', ps],
                       capture_output=True, text=True, errors='replace')
    if r.returncode:
        raise RuntimeError(f'PowerShell: {r.stderr.strip()[:600]}')


def by_wget(url, part, insecure):
    exe = shutil.which('wget')
    if not exe:
        raise Missing('wget not found')
    cmd = [exe, '-q', '-T', '60', '-O', part, url] + (['--no-check-certificate'] if insecure else [])
    r = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    if r.returncode:
        raise RuntimeError(f'wget {r.returncode}: {r.stderr.strip()[:600]}')


METHODS = [by_python, by_curl] + ([by_powershell] if WIN else [by_wget])


def download(url, out, sha=None, quiet=False):
    out = str(out)
    if sha and os.path.exists(out) and sha256(out) == sha:
        log(f'{os.path.basename(out)}: already present')
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    part = out + '.part'
    if not quiet:
        log(f'{os.path.basename(out)} <- {url.split("?")[0]}')
    for insecure in (False, True):
        if insecure:
            log('  WARNING: no secure connection (the network substitutes certificates: proxy or antivirus); '
                'retrying without certificate verification' + (', the file is checked by SHA-256' if sha else ''))
        for method in METHODS:
            for attempt in range(1, 4):
                try:
                    method(url, part, insecure)
                    if sha and sha256(part) != sha:
                        raise ValueError('checksum mismatch (file damaged or substituted in transit)')
                    os.replace(part, out)
                    return
                except Exception as e:  # noqa: BLE001
                    log(f'  {method.__name__[3:]}, attempt {attempt}: {" ".join(str(e).split())[:220]}')
                    if os.path.exists(part):
                        os.remove(part)
                    if isinstance(e, Missing) or any(w in str(e).upper() for w in CERT_WORDS):
                        break                    # no file, no program or certificate: next method
                    if isinstance(e, ValueError) and attempt == 2:
                        break
                    if attempt < 3:
                        time.sleep(10 * attempt)
    raise RuntimeError(f'could not download {url.split("?")[0]} by any method')


def release(key, out=None):
    a = ASSETS['release'][key]
    out = Path(out) if out else BUNDLE / a['dir'] / a['name']
    log(f'{a["name"]} ({a["size"] / 2 ** 20:.0f} MB) from {ASSETS["repo"]} release {ASSETS["tag"]}')
    download(f'https://github.com/{ASSETS["repo"]}/releases/download/{ASSETS["tag"]}/{a["name"]}', out, a['sha256'])
    return out


def yadisk_href(public_key, path):
    """Direct download link of one file of a public Yandex Disk folder (no account or token needed)."""
    q = urllib.parse.urlencode({'public_key': public_key, 'path': path})
    tmp = Path(os.environ.get('TMPDIR') or os.environ.get('TEMP') or '/tmp') / f'yadisk_{os.getpid()}.json'
    try:
        download(f'https://cloud-api.yandex.net/v1/disk/public/resources/download?{q}', tmp, quiet=True)
        return json.loads(tmp.read_text(encoding='utf-8'))['href']
    finally:
        if tmp.exists():
            tmp.unlink()


def dataset(out):
    d = ASSETS['dataset']
    out = Path(out)
    total = sum(f['size'] for f in d['files'])
    log(f'France dataset, {len(d["files"])} files, {total / 2 ** 30:.1f} GB, from {d["public_url"]}')
    for i, f in enumerate(d['files'], 1):
        dst = out / f['path'].lstrip('/')
        if dst.exists() and dst.stat().st_size == f['size'] and sha256(dst) == f['sha256']:
            log(f'[{i}/{len(d["files"])}] {f["path"]}: already present')
            continue
        log(f'[{i}/{len(d["files"])}] {f["path"]} ({f["size"] / 2 ** 20:.0f} MB)')
        for attempt in range(1, 4):              # the direct link is short-lived: resolve it again on failure
            try:
                download(yadisk_href(d['public_url'], f['path']), dst, f['sha256'], quiet=True)
                break
            except RuntimeError:
                if attempt == 3:
                    raise
                time.sleep(30)
    log(f'dataset ready: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('model', 'convnext', 'init', 'mmcv', 'dataset'))
    ap.add_argument('--out', help='target file (model, convnext, init, mmcv) or folder for the dataset')
    ap.add_argument('--windows', action='store_true', help='mmcv: the Windows build (win_amd64)')
    a = ap.parse_args()
    if a.what == 'dataset':
        dataset(a.out or BUNDLE / 'data' / 'France')
    elif a.what == 'mmcv' and a.windows:
        release('mmcv_windows', a.out)
    elif a.what == 'mmcv':
        try:
            release('mmcv', a.out)
        except RuntimeError as e:
            log(f'release unavailable ({e}); official OpenMMLab wheel')
            download(ASSETS['mmcv_official'], a.out or BUNDLE / 'wheels' / ASSETS['release']['mmcv']['name'])
    else:
        release(a.what, a.out)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print('FETCH FAILED:', e, file=sys.stderr)
        sys.exit(1)
