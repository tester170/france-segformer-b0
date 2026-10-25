# SegFormer-B0 для набора France, разметка IGN CoSIA (MMSegmentation 1.2.2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tester170/france-segformer-b0/blob/v1.0/France_SegFormer_B0.ipynb)

Семантическая сегментация ортофотоснимков Франции (около 0,2 м на пиксель) по 14 классам покрытия земли палитры IGN CoSIA моделью SegFormer-B0 в MMSegmentation 1.2.2: обученная модель, конфигурации MMSegmentation, подготовка данных, точное окружение и блокнот Colab для оценки, распознавания своих снимков и дообучения.

## Запуск в Colab
1. Откройте блокнот кнопкой «Open in Colab» выше.
2. *Среда выполнения → Сменить среду выполнения → GPU*.
3. *Среда выполнения → Выполнить все*.

Блокнот сам клонирует репозиторий, ставит окружение эксперимента (Python 3.11, PyTorch 2.1.0 + CUDA 12.1, MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation 1.2.2) отдельно от окружения Colab и скачивает модель из релиза `v1.0`. Свои снимки для распознавания загружаются кнопкой в разделе 2. Раздел 3 скачивает набор France с Яндекс Диска и проверяет модель штатным `tools/test.py`. Свои размеченные данные для дообучения загружаются архивом `my_data.zip` в разделе 4 (структура описана ниже), дообучение включается параметром `TRAIN` в разделе 5. Веса, маски и результаты скачиваются кнопками раздела 7.

## Автономная проверка на своём компьютере (Windows и Linux)
Скрипты `START_WINDOWS.bat` и `START_LINUX.sh` воспроизводят результат одним запуском, их достаточно скачать отдельно от репозитория: [START_WINDOWS.bat](https://raw.githubusercontent.com/tester170/france-segformer-b0/v1.0/START_WINDOWS.bat), [START_LINUX.sh](https://raw.githubusercontent.com/tester170/france-segformer-b0/v1.0/START_LINUX.sh). В Windows запуск — двойной щелчок по файлу, в Linux — `bash START_LINUX.sh`.

Скрипт ставит в папку `france_segformer_b0` рядом с собой uv, Python 3.11 и библиотеки с версиями эксперимента, скачивает код этого репозитория, веса модели и набор France с Яндекс Диска (5,4 ГБ, проверка SHA-256 каждого файла), готовит набор и запускает штатный `tools/test.py`. В конце выводится mIoU, ожидается 72,90 %. Система не меняется, всё удаляется вместе с папкой. Нужны около 20 ГБ на диске и интернет. Видеокарта NVIDIA желательна, без неё проверка идёт на процессоре, дольше. Повторный запуск продолжает с места остановки и ничего не скачивает заново.

Каждая загрузка выполняется с повторами и запасными способами: Python, curl, PowerShell или wget, зеркала PyPI. Если сеть подменяет сертификаты (корпоративный прокси, антивирус), скрипт сначала берёт сертификаты системы, затем загружает без проверки сертификата. Веса, MMCV и набор данных при этом всё равно сверяются по SHA-256.

## Свои данные для дообучения
```
my_data/
├── train/
│   ├── images/   снимки: GeoTIFF/TIFF, JP2, IMG, VRT, JPG, PNG, BMP, WEBP (8-битные RGB, ~0,2 м/пиксель)
│   └── masks/    маски с теми же именами: PNG или TIFF, коды классов CoSIA (255 — нет разметки) или цвета палитры CoSIA
└── val/          необязательно (иначе для проверки откладывается каждый пятый снимок)
    ├── images/
    └── masks/
```
Маска совпадает со своим снимком по размеру пиксель в пиксель. Коды классов — как в разметке CoSIA и в масках, которые пишет `scripts/infer.py` (таблица — в `MODEL_CARD_RU.md` и `splits/split_v1.json`). `scripts/prepare_own.py` раскладывает такие данные в структуру MMSegmentation.

## Без блокнота (Linux, GPU NVIDIA желательна)
```bash
git clone --branch v1.0 https://github.com/tester170/france-segformer-b0.git && cd france-segformer-b0
bash env/setup_env.sh ~/py311                                   # окружение Python 3.11 эксперимента
python3 scripts/fetch.py model                                  # веса модели из релиза -> checkpoints/
python3 scripts/fetch.py dataset --out data/France              # набор France с Яндекс Диска (5,4 ГБ)
~/py311/bin/python scripts/prepare_data.py --src data/France --out data/france
PYTHONPATH=. ~/py311/bin/python tools/test.py configs/france/segformer_mit-b0_1xb8-amp_france-1024x1024.py \
    checkpoints/segformer_b0_france_ep76.pth --work-dir work_dirs/test          # mIoU 72,90 %
~/py311/bin/python scripts/infer.py --input <снимок или папка> --out <папка> --overlay
```
Обучение с нуля, как в эксперименте:
```bash
python3 scripts/fetch.py init                                   # кодировщик MiT-B0, предобученный на LoveDA
PYTHONPATH=. ~/py311/bin/python tools/train.py configs/france/segformer_mit-b0_1xb8-amp_france-1024x1024.py \
    --work-dir work_dirs/segformer_b0_france
```
Подробнее о конфигурации — `configs/france/README_RU.md`. `tools/train.py` и `tools/test.py` — штатные скрипты MMSegmentation v1.2.2 без изменений.

## Набор данных
16 фрагментов ортофотоснимков (RGB GeoTIFF, около 12 600 × 9 400 пикселей, 0,2 м) с разметкой IGN CoSIA в трёх поднаборах: City (5 фрагментов), Village (7) и Other (4), всего 1,9 млрд пикселей. Набор лежит в публичной папке Яндекс Диска https://disk.yandex.ru/d/k1HfK1f8eh27Yw. `scripts/fetch.py dataset` скачивает его по прямым ссылкам без учётной записи и сверяет каждый файл с контрольной суммой SHA-256 из `env/assets.json`.

Обучение и проверка разделены по целым фрагментам (`splits/split_v1.json`): 12 обучающих и 4 валидационных (один из City, два из Village, один из Other, 25 % пикселей). Из 17 классов CoSIA в данных встречаются 14: снега, вырубки и класса «прочее» нет.

Разметка CoSIA создана нейросетевой сегментацией IGN, а не вручную, поэтому метрики показывают согласие модели с CoSIA.

## Файлы релиза `v1.0`
| Файл | Назначение |
|---|---|
| `segformer_b0_france_ep76.pth` | обученная модель (чекпойнт MMSegmentation 1.2.2, эпоха 76) |
| `mit_b0_loveda_encoder.pth` | кодировщик MiT-B0, предобученный на LoveDA: начальные веса для обучения |
| `mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl` | официальная сборка MMCV 2.1.0 OpenMMLab для torch 2.1 / CUDA 12.1 |
| `mmcv-2.1.0-cp311-cp311-win_amd64.whl` | сборка MMCV 2.1.0 с CUDA-операциями для Windows (torch 2.1 / CUDA 12.1) |

Контрольные суммы — в `env/assets.json`, `scripts/fetch.py` проверяет каждый скачанный файл.

## Модель
SegFormer-B0 (кодировщик MiT-B0 и MLP-декодер), 3,7 млн параметров, 14 классов, патчи 1024 × 1024, лучшая эпоха 76 из 100. Валидационные фрагменты: mIoU 72,9 %, F1 83,6 %, общая точность 88,6 %, κ = 0,860. По поднаборам общая точность 90,3 % (City), 87,9 % (Village) и 88,2 % (Other). Подробнее — `MODEL_CARD_RU.md`.

## Данные и лицензии
Классификация и разметка CoSIA (Couverture du Sol par Intelligence Artificielle) и ортофотоснимки подготовлены национальным институтом географической и лесной информации Франции (IGN) и распространяются им по открытой лицензии Etalab 2.0. Кодировщик предобучен на наборе LoveDA (J. Wang et al., *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation*, NeurIPS 2021). Модель SegFormer: E. Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.

## Состав
`configs/france/` конфигурации MMSegmentation · `france_mmseg/` модули проекта (потоковый скользящий инференс, повороты и отражения, управление скоростью обучения) · `tools/` штатные `train.py` и `test.py` MMSegmentation · `scripts/` загрузка файлов (`fetch.py`), подготовка набора France и своих данных, распознавание своих снимков · `env/` установка и проверка окружения, файлы релиза (`assets.json`) · `splits/` разбиение и таблица классов · `France_SegFormer_B0.ipynb` блокнот Colab · `START_WINDOWS.bat`, `START_LINUX.sh` автономная проверка на своём компьютере.
