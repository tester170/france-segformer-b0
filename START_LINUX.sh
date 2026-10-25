#!/usr/bin/env bash
# Воспроизведение модели SegFormer-B0 на наборе France (разметка IGN CoSIA) штатным tools/test.py MMSegmentation 1.2.2.
# Всё ставится и скачивается в папку france_segformer_b0 рядом со скриптом, система не меняется: uv и Python 3.11,
# окружение с версиями эксперимента (PyTorch 2.1.0 + CUDA 12.1, MMCV 2.1.0, MMSegmentation 1.2.2), код и конфигурация
# из репозитория tester170/france-segformer-b0 (v1.0), веса модели, набор France с Яндекс Диска (5,4 ГБ).
# Повторный запуск ничего не скачивает заново. Нужны Linux x86_64, около 20 ГБ на диске, интернет; видеокарта NVIDIA
# с драйвером (не ниже 525) желательна, без неё проверка идёт на процессоре (дольше).
# Каждая загрузка идёт с повторами и запасными способами (curl, wget, Python; зеркала PyPI). Если сеть подменяет
# сертификаты (корпоративный прокси, антивирус), сначала берутся сертификаты системы, затем загрузка без проверки
# сертификата; веса, MMCV и набор данных при этом проверяются по SHA-256.
# Запуск: bash START_LINUX.sh
set -euo pipefail
cd -- "$(dirname -- "$0")"
SELF="$PWD/$(basename -- "$0")"
# всё, что скачивается и ставится, лежит в одной папке рядом со скриптом
mkdir -p france_segformer_b0
cd france_segformer_b0
ROOT="$PWD"
REPO=tester170/france-segformer-b0
TAG=v1.0
CONFIG=configs/france/segformer_mit-b0_1xb8-amp_france-1024x1024.py
CKPT=checkpoints/segformer_b0_france_ep76.pth
TORCH_INDEX=https://download.pytorch.org/whl/cu121
INDEXES="https://pypi.org/simple https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple https://repo.huaweicloud.com/repository/pypi/simple"
INSECURE_HOSTS="pypi.org files.pythonhosted.org download.pytorch.org download-r2.pytorch.org mirrors.tuna.tsinghua.edu.cn repo.huaweicloud.com github.com objects.githubusercontent.com release-assets.githubusercontent.com"
export UV_UNMANAGED_INSTALL="$ROOT/.runtime/uv" UV_PYTHON_INSTALL_DIR="$ROOT/.runtime/python" \
       UV_CACHE_DIR="$ROOT/.runtime/cache/uv" TMPDIR="$ROOT/.runtime/tmp" UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 \
       PIP_CONFIG_FILE=/dev/null PIP_DEFAULT_TIMEOUT=120 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p "$TMPDIR"
step() { echo; echo "=== $*"; }
fail() { echo; echo "ОШИБКА: $*"; exit 1; }
warn_insecure() { echo "  ВНИМАНИЕ: защищённое соединение не установилось (сеть подменяет сертификаты: прокси или антивирус), повтор без проверки сертификата"; }

# файл по адресу: curl, затем wget; $3=k: без проверки сертификата
fetch() {
  if command -v curl >/dev/null 2>&1; then
    curl -fLsS --retry 3 --retry-delay 5 --connect-timeout 20 ${3:+--insecure} "$1" -o "$2" && return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q -T 60 -t 3 ${3:+--no-check-certificate} -O "$2" "$1" && return 0
  fi
  return 1
}

# команда uv: обычная проверка сертификатов, сертификаты системы, затем без проверки; при сбоях сети до 3 кругов
tls_retry() {
  local try
  for try in 1 2 3; do
    "$@" && return 0
    echo "  повтор с сертификатами системы"
    UV_SYSTEM_CERTS=1 UV_NATIVE_TLS=1 "$@" && return 0
    warn_insecure
    UV_SYSTEM_CERTS=1 UV_NATIVE_TLS=1 UV_INSECURE_HOST="$INSECURE_HOSTS" "$@" && return 0
    [ "$try" = 3 ] && return 1
    echo "  сеть недоступна, повтор через 20 с (круг $try/3)"; sleep 20
  done
}

# библиотеки по списку: uv с зеркалами PyPI (три способа проверки сертификатов), затем pip
pkg_install() {
  local mode idx host trusted=()
  for host in $INSECURE_HOSTS; do trusted+=(--trusted-host "$host"); done
  for mode in default system insecure; do
    [ "$mode" = insecure ] && warn_insecure
    for idx in $INDEXES; do
      ( case $mode in
          system) export UV_SYSTEM_CERTS=1 UV_NATIVE_TLS=1 ;;
          insecure) export UV_SYSTEM_CERTS=1 UV_NATIVE_TLS=1 UV_INSECURE_HOST="$INSECURE_HOSTS" ;;
        esac
        "$UV" --no-config pip install --python "$PY" --no-deps --index-strategy unsafe-best-match \
          --index-url "$idx" --extra-index-url "$TORCH_INDEX" "$@" ) && return 0
      echo "  не удалось через $idx, следующий вариант"
    done
    if "$PY" -m pip --version >/dev/null 2>&1; then          # pip берёт сертификаты системы
      case $mode in
        system) "$PY" -m pip install --no-deps --extra-index-url "$TORCH_INDEX" "$@" && return 0 ;;
        insecure) "$PY" -m pip install --no-deps --extra-index-url "$TORCH_INDEX" "${trusted[@]}" "$@" && return 0 ;;
      esac
    fi
  done
  return 1
}

if ! { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
  echo "ПРЕДУПРЕЖДЕНИЕ: видеокарта NVIDIA не найдена, проверка модели пойдёт на процессоре."
  echo "                Это займёт заметно больше времени (десятки минут вместо минуты), результат тот же."
fi

step "1/8 uv и Python 3.11"
install_uv() {
  local try url arch dir
  arch="$(uname -m)"; dir="uv-$arch-unknown-linux-gnu"
  for try in 1 2 3 4 5; do
    # установщик uv: astral.sh, затем GitHub
    for url in https://astral.sh/uv/install.sh https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh; do
      fetch "$url" "$TMPDIR/install-uv.sh" && INSTALLER_NO_MODIFY_PATH=1 sh "$TMPDIR/install-uv.sh" >/dev/null 2>&1 \
        && [ -x .runtime/uv/uv ] && return 0
    done
    # архив uv напрямую с GitHub, при подмене сертификатов без проверки
    url="https://github.com/astral-sh/uv/releases/latest/download/$dir.tar.gz"
    if fetch "$url" "$TMPDIR/uv.tar.gz" || { warn_insecure; fetch "$url" "$TMPDIR/uv.tar.gz" k; }; then
      rm -rf "${TMPDIR:?}/$dir"
      tar -xzf "$TMPDIR/uv.tar.gz" -C "$TMPDIR" && mkdir -p .runtime/uv && cp "$TMPDIR/$dir/uv" .runtime/uv/uv \
        && chmod +x .runtime/uv/uv && return 0
    fi
    [ "$try" = 5 ] && return 1
    echo "  сеть недоступна, повтор через 20 с (попытка $try/5)"; sleep 20
  done
}
[ -x .runtime/uv/uv ] || install_uv || fail "не удалось скачать uv, проверьте подключение к интернету"
UV="$ROOT/.runtime/uv/uv"
tls_retry "$UV" --no-config python install 3.11 --no-bin || fail "не удалось установить Python 3.11"
if [ ! -x .venv/bin/python ]; then
  tls_retry "$UV" --no-config venv --clear --python 3.11 --seed .venv \
    || "$UV" --no-config venv --clear --python 3.11 .venv || fail "не удалось создать окружение .venv"
fi
PY="$ROOT/.venv/bin/python"

# загрузчик (Python, curl, wget; SHA-256): его код в конце этого файла
HELPER="$ROOT/.runtime/helper.py"
"$PY" -I -c "import sys; t = open(sys.argv[1], encoding='utf-8').read(); open(sys.argv[2], 'w', encoding='utf-8').write(t.split('\n#__HELPER_PY__\n', 1)[1].split('\n__HELPER_PY_END__', 1)[0] + '\n')" "$SELF" "$HELPER"
H() { "$PY" -I "$HELPER" "$@"; }

step "2/8 код и конфигурация ($REPO, $TAG)"
H github-zip "https://github.com/$REPO/archive/refs/tags/$TAG.zip" code || fail "код проекта не скачан"

step "3/8 библиотеки с версиями эксперимента (PyTorch 2.1.0 + CUDA 12.1, MMSegmentation 1.2.2): несколько минут"
if [ ! -f .venv/.packages_ok ]; then
  pkg_install -r code/env/requirements.lock || fail "не удалось установить библиотеки"
  touch .venv/.packages_ok
fi

step "4/8 MMCV 2.1.0 (сборка OpenMMLab для PyTorch 2.1 / CUDA 12.1)"
if ! "$PY" -c "import mmcv, mmcv.ops" >/dev/null 2>&1; then
  WHL="$TMPDIR/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl"
  H mmcv code/env/assets.json "$WHL" || fail "MMCV не скачан"
  "$UV" --no-config pip install --python "$PY" --no-deps "$WHL" || "$PY" -m pip install --no-deps "$WHL" \
    || fail "MMCV не установлен"
fi

step "5/8 веса модели (15 МБ, проверка SHA-256)"
H asset code/env/assets.json model "code/$CKPT" || fail "веса модели не скачаны"

step "6/8 набор France с Яндекс Диска (32 файла, 5,4 ГБ, проверка SHA-256 каждого файла)"
H yadisk code/env/assets.json data/France || fail "набор данных не скачан"

step "7/8 подготовка набора в структуре MMSegmentation (около 2 минут)"
cd code
"$PY" scripts/prepare_data.py --src ../data/France --out data/france --workers 3 || fail "набор не подготовлен"

step "8/8 проверка модели: tools/test.py MMSegmentation на валидационных фрагментах"
PYTHONPATH=. MPLBACKEND=Agg "$PY" tools/test.py "$CONFIG" "$CKPT" --work-dir work_dirs/test 2>&1 | tee ../test.log \
  | grep -E "^\||Iter\(test\)|Traceback|Error:" || true
"$PY" - <<'EOF'
import glob, json
logs = sorted(glob.glob('work_dirs/test/*/*.json'))
if not logs:
    raise SystemExit('\nОШИБКА: tools/test.py не завершился, подробности в test.log')
m = json.loads(open(logs[-1]).read().strip().splitlines()[-1])
print(f"\nГОТОВО. mIoU {m['mIoU']:.2f} % (ожидается 72.90), mDice {m['mDice']:.2f} %, OA {m['aAcc']:.2f} %")
EOF
exit 0
# код загрузчика (извлекается в .runtime/helper.py, bash его не выполняет)
: <<'__HELPER_PY_END__'
#__HELPER_PY__
# Загрузчик START_LINUX.sh / START_WINDOWS.bat (только стандартная библиотека Python; скрипт извлекает его из себя).
# Каждый файл пробуется по цепочке способов: Python, curl, PowerShell (Windows) или wget (Linux), каждый с повторами
# при сбоях сети. Если защищённое соединение не устанавливается ни одним способом (сеть подменяет сертификаты:
# корпоративный прокси, антивирус), загрузка повторяется без проверки сертификата. Веса модели, MMCV и набор данных
# при этом защищены проверкой SHA-256, для остальных файлов выводится предупреждение.
#   get URL OUT [SHA256]           файл
#   github-zip URL DIR             архив GitHub (одна папка верхнего уровня) -> её содержимое в DIR
#   asset ASSETS_JSON KEY OUT      файл релиза из env/assets.json (адрес и SHA-256 оттуда)
#   mmcv ASSETS_JSON OUT           MMCV для Linux: релиз проекта, затем сервер OpenMMLab
#   yadisk ASSETS_JSON OUTDIR      набор France с Яндекс Диска (прямые ссылки по публичной ссылке, SHA-256)
import hashlib
import io
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

WIN = os.name == 'nt'
UA = {'User-Agent': 'skyscapes-reproduce'}
SYSTEM_CA = ['/etc/ssl/certs/ca-certificates.crt', '/etc/pki/tls/certs/ca-bundle.crt', '/etc/ssl/ca-bundle.pem',
             '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem', '/etc/ssl/cert.pem']


def log(msg):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


class Missing(Exception):
    """Файла нет на сервере или способ недоступен: повторять этот способ бессмысленно."""


def by_python(url, part, insecure):
    if insecure:
        ctx = ssl._create_unverified_context()
    else:
        ctx = ssl.create_default_context()
        for ca in SYSTEM_CA:                     # сертификаты системы, включая добавленные администратором
            if os.path.exists(ca):
                try:
                    ctx.load_verify_locations(ca)
                except (OSError, ssl.SSLError):
                    pass
    req = urllib.request.Request(url, headers=UA)
    try:
        r = urllib.request.urlopen(req, timeout=120, context=ctx)
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500:
            raise Missing(f'HTTP {e.code}') from e
        raise
    with r, open(part, 'wb') as f:
        total, done, shown = int(r.headers.get('Content-Length') or 0), 0, -10
        for chunk in iter(lambda: r.read(1 << 22), b''):
            f.write(chunk)
            done += len(chunk)
            pct = 100 * done // total if total else 0
            if total > 20 * 2 ** 20 and (pct >= shown + 10 or done == total):
                shown = pct
                log(f'  {done / 2 ** 20:.0f}/{total / 2 ** 20:.0f} МБ')


def curl_exe():
    if WIN:
        exe = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'curl.exe')
        if os.path.exists(exe):
            return exe
    return shutil.which('curl')


def by_curl(url, part, insecure):
    exe = curl_exe()
    if not exe:
        raise Missing('curl не найден')
    cmd = [exe, '--fail', '--location', '--silent', '--show-error', '--connect-timeout', '30', '--output', part, url]
    if insecure:
        cmd.insert(1, '--insecure')
    r = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    if r.returncode and WIN and not insecure:    # Windows: сервер отзыва сертификатов часто закрыт в корпоративных сетях
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
        raise Missing('wget не найден')
    cmd = [exe, '-q', '-T', '60', '-O', part, url] + (['--no-check-certificate'] if insecure else [])
    r = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    if r.returncode:
        raise RuntimeError(f'wget {r.returncode}: {r.stderr.strip()[:600]}')


METHODS = [by_python, by_curl] + ([by_powershell] if WIN else [by_wget])
CERT_WORDS = ('CERTIFICATE', 'SSL', 'TLS', 'SCHANNEL', 'CURL 60', 'CURL 35', 'CURL 77', 'TRUST', 'CERT_',
              'СЕРТИФИКАТ', 'ДОВЕРИТЕЛЬН', 'ЗАЩИЩЕННОГО КАНАЛА', 'ЗАЩИЩЁННОГО КАНАЛА')


def download(url, out, sha=None):
    if os.path.exists(out) and (sha is None or sha256(out) == sha):
        log(f'{os.path.basename(out)}: уже скачан')
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    part = out + '.part'
    log(f'{os.path.basename(out)} <- {url.split("?")[0][:80]}')
    for insecure in (False, True):
        if insecure:
            log('  ВНИМАНИЕ: защищённое соединение не установилось (сеть подменяет сертификаты: прокси или антивирус).')
            log('  Повтор без проверки сертификата, ' + ('целостность файла проверяется по SHA-256.' if sha else
                'у этого файла нет контрольной суммы.'))
        for method in METHODS:
            for attempt in range(1, 4):
                try:
                    method(url, part, insecure)
                    if sha and sha256(part) != sha:
                        raise ValueError('контрольная сумма не совпала (файл повреждён или подменён при загрузке)')
                    os.replace(part, out)
                    return
                except Exception as e:  # noqa: BLE001
                    log(f'  {method.__name__[3:]}, попытка {attempt}: {" ".join(str(e).split())[:220]}')
                    if os.path.exists(part):
                        os.remove(part)
                    if isinstance(e, Missing) or any(w in str(e).upper() for w in CERT_WORDS):
                        break                    # нет файла, нет программы или сертификат: следующий способ
                    if isinstance(e, ValueError) and attempt == 2:
                        break                    # файл дважды пришёл повреждённым: следующий способ
                    if attempt < 3:
                        time.sleep(10 * attempt)
    raise RuntimeError(f'не удалось скачать {url} ни одним способом')


def github_zip(url, dest):
    if os.path.exists(os.path.join(dest, '.ok')):
        log(f'{dest}: уже есть')
        return
    z = dest.rstrip('/\\') + '.zip'
    download(url, z)
    tmp = dest.rstrip('/\\') + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp)
    (top,) = os.listdir(tmp)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    os.replace(os.path.join(tmp, top), dest)
    shutil.rmtree(tmp)
    os.remove(z)
    open(os.path.join(dest, '.ok'), 'w').close()


def asset(assets_json, key, out):
    a = json.load(open(assets_json, encoding='utf-8'))
    r = a['release'][key]
    download(f'https://github.com/{a["repo"]}/releases/download/{a["tag"]}/{r["name"]}', out, r['sha256'])


def mmcv(assets_json, out):
    try:
        asset(assets_json, 'mmcv', out)
    except Exception as e:  # noqa: BLE001
        log(f'релиз проекта недоступен ({e}), сборка OpenMMLab')
        download(json.load(open(assets_json, encoding='utf-8'))['mmcv_official'], out)


def yadisk(assets_json, outdir):
    """Набор France из публичной папки Яндекс Диска: по каждому файлу открытый сервис Диска выдаёт прямую ссылку
    (без учётной записи и токена), файл скачивается и сверяется с SHA-256 из assets.json."""
    import urllib.parse
    d = json.load(open(assets_json, encoding='utf-8'))['dataset']
    n = len(d['files'])
    for i, f in enumerate(d['files'], 1):
        dst = os.path.join(outdir, *f['path'].strip('/').split('/'))
        if os.path.exists(dst) and os.path.getsize(dst) == f['size'] and sha256(dst) == f['sha256']:
            log(f'[{i}/{n}] {f["path"]}: уже скачан')
            continue
        log(f'[{i}/{n}] {f["path"]} ({f["size"] / 2 ** 20:.0f} МБ)')
        q = urllib.parse.urlencode({'public_key': d['public_url'], 'path': f['path']})
        meta = dst + '.link.json'
        for attempt in range(1, 4):              # прямая ссылка живёт недолго: при сбое запрашивается заново
            try:
                if os.path.exists(meta):
                    os.remove(meta)
                download(f'https://cloud-api.yandex.net/v1/disk/public/resources/download?{q}', meta)
                href = json.load(open(meta, encoding='utf-8'))['href']
                os.remove(meta)
                download(href, dst, f['sha256'])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                log(f'  повтор через 30 с: {e}')
                time.sleep(30)
    log(f'набор готов: {outdir}')


COMMANDS = {'get': lambda url, out, sha=None: download(url, out, sha), 'github-zip': github_zip, 'asset': asset,
            'mmcv': mmcv, 'yadisk': yadisk}

if __name__ == '__main__':
    try:
        COMMANDS[sys.argv[1]](*sys.argv[2:])
    except Exception as e:  # noqa: BLE001
        sys.exit(f'ОШИБКА: {e}. Проверьте подключение к интернету и запустите скрипт ещё раз.')
__HELPER_PY_END__
