@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
REM всё, что скачивается и ставится, лежит в одной папке рядом со скриптом
if not exist "%~dp0france_segformer_b0" mkdir "%~dp0france_segformer_b0"
cd /d "%~dp0france_segformer_b0" || exit /b 1
set "ROOT=%CD%"

REM Воспроизведение модели SegFormer-B0 на наборе France (разметка IGN CoSIA) штатным tools/test.py MMSegmentation 1.2.2.
REM Всё ставится и скачивается в папку france_segformer_b0 рядом со скриптом, система не меняется: uv и Python 3.11,
REM окружение с версиями эксперимента (PyTorch 2.1.0 + CUDA 12.1, MMCV 2.1.0, MMSegmentation 1.2.2), код и конфигурация
REM из репозитория tester170/france-segformer-b0 (v1.0), веса модели, набор France с Яндекс Диска (5,4 ГБ).
REM Повторный запуск ничего не скачивает заново.
REM Нужны Windows 10/11 x64, около 20 ГБ на диске, интернет; видеокарта NVIDIA с драйвером (не ниже 531) желательна,
REM без неё проверка идёт на процессоре (дольше).
REM Каждая загрузка идёт с повторами и запасными способами (PowerShell, curl, Python; зеркала PyPI). Если сеть подменяет
REM сертификаты (корпоративный прокси, антивирус), сначала берутся сертификаты Windows, затем загрузка без проверки
REM сертификата; веса, MMCV и набор данных при этом проверяются по SHA-256.
set "REPO=tester170/france-segformer-b0"
set "TAG=v1.0"
set "CONFIG=configs/france/segformer_mit-b0_1xb8-amp_france-1024x1024.py"
set "CKPT=checkpoints/segformer_b0_france_ep76.pth"
set "MMCV_WHEEL=mmcv-2.1.0-cp311-cp311-win_amd64.whl"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
set "INDEXES=https://pypi.org/simple https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple https://repo.huaweicloud.com/repository/pypi/simple"
set "INSECURE_HOSTS=pypi.org files.pythonhosted.org download.pytorch.org download-r2.pytorch.org mirrors.tuna.tsinghua.edu.cn repo.huaweicloud.com github.com objects.githubusercontent.com release-assets.githubusercontent.com"
set "TRUSTED=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --trusted-host mirrors.tuna.tsinghua.edu.cn --trusted-host repo.huaweicloud.com"

set "UV_UNMANAGED_INSTALL=%ROOT%\.runtime\uv"
set "UV_PYTHON_INSTALL_DIR=%ROOT%\.runtime\python"
set "UV_PYTHON_NO_REGISTRY=1"
set "UV_CACHE_DIR=%ROOT%\.runtime\cache\uv"
set "UV_HTTP_TIMEOUT=300"
set "UV_HTTP_RETRIES=5"
set "UV_SYSTEM_CERTS="
set "UV_NATIVE_TLS="
set "UV_INSECURE_HOST="
set "PIP_CONFIG_FILE=NUL"
set "PIP_DEFAULT_TIMEOUT=120"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TEMP=%ROOT%\.runtime\tmp"
set "TMP=%TEMP%"
set "PSModulePath=%SystemRoot%\System32\WindowsPowerShell\v1.0\Modules"
if not exist "%TEMP%" mkdir "%TEMP%"
set "CURL=%SystemRoot%\System32\curl.exe"
set "PS=powershell.exe -NoProfile -ExecutionPolicy Bypass -Command"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "UV=%ROOT%\.runtime\uv\uv.exe"
set "HELPER=%ROOT%\.runtime\helper.py"

set "GPU_OK=1"
where nvidia-smi >nul 2>&1 || set "GPU_OK=0"
if "%GPU_OK%"=="1" nvidia-smi -L >nul 2>&1 || set "GPU_OK=0"
if "%GPU_OK%"=="0" (
    echo ПРЕДУПРЕЖДЕНИЕ: видеокарта NVIDIA не найдена, проверка модели пойдёт на процессоре.
    echo                 Это займёт заметно больше времени ^(десятки минут вместо минуты^), результат тот же.
)

echo.
echo === 1/8 uv и Python 3.11
if exist "%UV%" goto UV_READY
set /a "UV_TRY=0"
:UV_DOWNLOAD
set /a "UV_TRY+=1"
REM 1) установщик uv (GitHub, astral.sh) через PowerShell и curl
%PS% "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing https://github.com/astral-sh/uv/releases/latest/download/uv-installer.ps1 -TimeoutSec 120 -OutFile '%TEMP%\uv-install.ps1'" >nul 2>&1 && call :UV_SETUP && goto UV_READY
"%CURL%" --fail --location --silent --show-error --ssl-revoke-best-effort --retry 3 --connect-timeout 20 https://astral.sh/uv/install.ps1 --output "%TEMP%\uv-install.ps1" && call :UV_SETUP && goto UV_READY
REM 2) архив uv напрямую с GitHub: curl, PowerShell, при подмене сертификатов без проверки
set "UV_ZIP_URL=https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
"%CURL%" --fail --location --silent --show-error --ssl-revoke-best-effort --retry 3 --connect-timeout 20 "%UV_ZIP_URL%" --output "%TEMP%\uv.zip" && call :UV_UNZIP && goto UV_READY
%PS% "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing '%UV_ZIP_URL%' -TimeoutSec 300 -OutFile '%TEMP%\uv.zip'" >nul 2>&1 && call :UV_UNZIP && goto UV_READY
call :WARN_INSECURE
"%CURL%" --insecure --fail --location --silent --show-error --retry 3 --connect-timeout 20 "%UV_ZIP_URL%" --output "%TEMP%\uv.zip" && call :UV_UNZIP && goto UV_READY
%PS% "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; [Net.ServicePointManager]::ServerCertificateValidationCallback={$true}; Invoke-WebRequest -UseBasicParsing '%UV_ZIP_URL%' -TimeoutSec 300 -OutFile '%TEMP%\uv.zip'" >nul 2>&1 && call :UV_UNZIP && goto UV_READY
if %UV_TRY% geq 5 (echo ОШИБКА: не удалось скачать uv, проверьте подключение к интернету. & goto FAIL)
echo   сеть недоступна, повтор через 20 с ^(попытка %UV_TRY%/5^)
ping -n 21 127.0.0.1 >nul
goto UV_DOWNLOAD
:UV_READY
call :TLS "%UV%" --no-config python install 3.11 --no-bin || (echo ОШИБКА: не удалось установить Python 3.11 & goto FAIL)
if exist "%PY%" goto VENV_READY
call :TLS "%UV%" --no-config venv --clear --python 3.11 --seed .venv && goto VENV_READY
"%UV%" --no-config venv --clear --python 3.11 .venv || (echo ОШИБКА: не удалось создать окружение .venv & goto FAIL)
:VENV_READY
REM загрузчик (Python, curl, PowerShell; SHA-256): его код в конце этого файла
"%PY%" -I -c "import sys; t = open(sys.argv[1], encoding='utf-8').read(); open(sys.argv[2], 'w', encoding='utf-8').write(t.split('\n#__HELPER_PY__\n', 1)[1].split('\n__HELPER_PY_END__', 1)[0] + '\n')" "%~f0" "%HELPER%" || goto FAIL

echo.
echo === 2/8 код и конфигурация (%REPO%, %TAG%)
"%PY%" -I "%HELPER%" github-zip "https://github.com/%REPO%/archive/refs/tags/%TAG%.zip" code || goto FAIL

echo.
echo === 3/8 библиотеки с версиями эксперимента (PyTorch 2.1.0 + CUDA 12.1, MMSegmentation 1.2.2): несколько минут
if exist ".venv\packages.ok" goto PACKAGES_READY
REM triton: только для Linux (в PyTorch 2.1 под Windows не используется)
findstr /v /b /c:"triton==" "code\env\requirements.lock" > "%TEMP%\requirements_windows.txt"
call :PKG -r "%TEMP%\requirements_windows.txt" || (echo ОШИБКА: не удалось установить библиотеки & goto FAIL)
type nul > ".venv\packages.ok"
:PACKAGES_READY

echo.
echo === 4/8 MMCV 2.1.0 с CUDA-операциями (сборка для Windows, PyTorch 2.1 / CUDA 12.1)
"%PY%" -c "import mmcv.ops" >nul 2>&1
if errorlevel 1 (
    "%PY%" -I "%HELPER%" asset code\env\assets.json mmcv_windows "%TEMP%\%MMCV_WHEEL%" || goto FAIL
    "%UV%" --no-config pip install --python "%PY%" --no-deps "%TEMP%\%MMCV_WHEEL%" || "%PY%" -m pip install --no-deps "%TEMP%\%MMCV_WHEEL%" || goto FAIL
)
"%PY%" -c "import torch, mmcv, mmseg; from mmcv.ops import nms; print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmseg', mmseg.__version__, '|', torch.cuda.get_device_name() if torch.cuda.is_available() else 'видеокарта не найдена: расчёт на процессоре')" || goto FAIL

echo.
echo === 5/8 веса модели (15 МБ, проверка SHA-256)
"%PY%" -I "%HELPER%" asset code\env\assets.json model "code\%CKPT%" || goto FAIL

echo.
echo === 6/8 набор France с Яндекс Диска (32 файла, 5,4 ГБ, проверка SHA-256 каждого файла)
"%PY%" -I "%HELPER%" yadisk code\env\assets.json data\France || goto FAIL

echo.
echo === 7/8 подготовка набора в структуре MMSegmentation (около 2 минут)
pushd code
"%PY%" scripts\prepare_data.py --src ..\data\France --out data\france --workers 3
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" goto FAIL

echo.
echo === 8/8 проверка модели: tools/test.py MMSegmentation на валидационных фрагментах
pushd code
set "PYTHONPATH=%ROOT%\code"
set "MPLBACKEND=Agg"
"%PY%" tools\test.py %CONFIG% %CKPT% --work-dir work_dirs\test
"%PY%" -c "import glob, json, sys; logs = sorted(glob.glob('work_dirs/test/*/*.json')); logs or sys.exit('ОШИБКА: tools/test.py не завершился'); m = json.loads(open(logs[-1]).read().strip().splitlines()[-1]); print('\nГОТОВО. mIoU %%.2f %%%% (ожидается 72.90), mDice %%.2f %%%%, OA %%.2f %%%%' %% (m['mIoU'], m['mDice'], m['aAcc']))"
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" goto FAIL
pause
exit /b 0

:UV_SETUP
%PS% "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $env:INSTALLER_NO_MODIFY_PATH='1'; & '%TEMP%\uv-install.ps1'" >nul 2>&1
if exist "%UV%" exit /b 0
exit /b 1

:UV_UNZIP
if exist "%TEMP%\uv_zip" rmdir /s /q "%TEMP%\uv_zip"
mkdir "%TEMP%\uv_zip"
tar -xf "%TEMP%\uv.zip" -C "%TEMP%\uv_zip" || exit /b 1
if not exist "%UV_UNMANAGED_INSTALL%" mkdir "%UV_UNMANAGED_INSTALL%"
for /r "%TEMP%\uv_zip" %%F in (uv.exe uvx.exe uvw.exe) do if exist "%%F" copy /y "%%F" "%UV_UNMANAGED_INSTALL%\" >nul
if exist "%UV%" exit /b 0
exit /b 1

:WARN_INSECURE
echo   ВНИМАНИЕ: защищённое соединение не установилось ^(сеть подменяет сертификаты: прокси или антивирус^), повтор без проверки сертификата
exit /b 0

:TLS
REM команда uv: обычная проверка сертификатов, сертификаты Windows, затем без проверки; при сбоях сети до 3 кругов
setlocal
set /a "TLS_TRY=0"
:TLS_AGAIN
set /a "TLS_TRY+=1"
set "UV_SYSTEM_CERTS="
set "UV_NATIVE_TLS="
set "UV_INSECURE_HOST="
%* && (endlocal & exit /b 0)
echo   повтор с сертификатами Windows
set "UV_SYSTEM_CERTS=1"
set "UV_NATIVE_TLS=1"
%* && (endlocal & exit /b 0)
call :WARN_INSECURE
set "UV_INSECURE_HOST=%INSECURE_HOSTS%"
%* && (endlocal & exit /b 0)
if %TLS_TRY% geq 3 (endlocal & exit /b 1)
echo   сеть недоступна, повтор через 20 с ^(круг %TLS_TRY%/3^)
ping -n 21 127.0.0.1 >nul
goto TLS_AGAIN

:PKG
REM библиотеки по списку: uv с зеркалами PyPI (три способа проверки сертификатов), затем pip
setlocal
set "UV_SYSTEM_CERTS="
set "UV_NATIVE_TLS="
set "UV_INSECURE_HOST="
for %%I in (%INDEXES%) do (
    "%UV%" --no-config pip install --python "%PY%" --no-deps --index-strategy unsafe-best-match --index-url %%I --extra-index-url %TORCH_INDEX% %* && (endlocal & exit /b 0)
    echo   не удалось через %%I, следующий вариант
)
set "UV_SYSTEM_CERTS=1"
set "UV_NATIVE_TLS=1"
for %%I in (%INDEXES%) do (
    "%UV%" --no-config pip install --python "%PY%" --no-deps --index-strategy unsafe-best-match --index-url %%I --extra-index-url %TORCH_INDEX% %* && (endlocal & exit /b 0)
    echo   не удалось через %%I, следующий вариант
)
REM pip берёт сертификаты Windows
"%PY%" -m pip install --no-deps --extra-index-url %TORCH_INDEX% %* && (endlocal & exit /b 0)
call :WARN_INSECURE
set "UV_INSECURE_HOST=%INSECURE_HOSTS%"
for %%I in (%INDEXES%) do (
    "%UV%" --no-config pip install --python "%PY%" --no-deps --index-strategy unsafe-best-match --index-url %%I --extra-index-url %TORCH_INDEX% %* && (endlocal & exit /b 0)
    echo   не удалось через %%I, следующий вариант
)
"%PY%" -m pip install --no-deps --extra-index-url %TORCH_INDEX% %TRUSTED% %* && (endlocal & exit /b 0)
endlocal & exit /b 1

:FAIL
echo.
echo Не удалось выполнить проверку. Прочитайте сообщение выше.
pause
exit /b 1
REM код загрузчика (извлекается в .runtime\helper.py, cmd его не выполняет)
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
