# Конфигурации MMSegmentation для моделей набора France

Конфигурации трёх моделей отчёта в формате MMSegmentation 1.2.2 (наследование от `_base_`, штатные `BaseSegDataset`, `IoUMetric`, преобразования MMSegmentation) запускаются штатными `tools/train.py` и `tools/test.py`.

| Конфигурация | Модель | Веса | mIoU (val, 14 классов) |
|---|---|---|---|
| `segformer_mit-b0_1xb8-amp_france-1024x1024.py` | SegFormer-B0 | `segformer_b0_france_ep76.pth` ([релиз](https://github.com/tester170/france-segformer-b0/releases/tag/v1.0)) | 72,90 |
| `dinov3-convnext-b_upernet_1xb2-amp_france-1024x1024.py` | DINOv3 ConvNeXt-B + UPerNet | `convnext_b_upernet_france_ep43.pth` (тот же релиз) | 75,30 |
| `dinov3-vitl16-adapter_upernet_1xb2-amp_france-1024x1024.py` | DINOv3 ViT-L/16 + адаптер + UPerNet (собственная) | `vitl16_adapter_upernet_france.pth` (тот же релиз) | 80,35\* |

\* Веса ViT-L + UPerNet получены дообучением на 84 дополнительных фрагментах разметки IGN CoSIA из других регионов Франции, которые не входят в опубликованный набор (всего 96 фрагментов). После обучения по этой конфигурации на 12 фрагментах набора, то есть при тех же данных, что у двух других моделей, в эксперименте получено 78,67 %.

Значения mIoU получены штатным `tools/test.py` с этими конфигурациями и весами и совпадают с отчётами. Все модели обучены только на данных France: ConvNeXt-B + UPerNet и ViT-L + UPerNet начинали с исходных весов DINOv3, а не с моделей, обученных на DLR SkyScapes.

## Подготовка
```bash
bash env/setup_env.sh ~/py311                         # Python 3.11, PyTorch 2.1.0, MMCV 2.1.0, MMSegmentation 1.2.2
python3 scripts/fetch.py dataset --out data/France    # набор France с Яндекс Диска, проверка SHA-256
~/py311/bin/python scripts/prepare_data.py --src data/France --out data/france
python3 scripts/fetch.py model                        # SegFormer-B0 -> checkpoints/
python3 scripts/fetch.py convnext                     # ConvNeXt-B + UPerNet -> checkpoints/ (466 МБ)
python3 scripts/fetch.py vitl                         # ViT-L + UPerNet -> checkpoints/ (1,3 ГБ)
bash env/fetch_vendor.sh                              # официальный код DINOv3 (для ConvNeXt-B и ViT-L) -> vendor/dinov3
```
`prepare_data.py` раскладывает набор в структуру MMSegmentation: `data/france/img_dir|ann_dir/train` — плитки 2048 × 2048 двенадцати обучающих фрагментов, `img_dir|ann_dir/val` — четыре валидационных фрагмента целиком. Метки переводятся из кодов CoSIA (0–5, 7–14) в номера классов 0–13. Команды ниже запускаются из корня репозитория с `PYTHONPATH=.`: так MMSegmentation находит модули `france_mmseg`, перечисленные в `custom_imports`.

## Оценка
```bash
PYTHONPATH=. ~/py311/bin/python tools/test.py configs/france/segformer_mit-b0_1xb8-amp_france-1024x1024.py \
    checkpoints/segformer_b0_france_ep76.pth
PYTHONPATH=. ~/py311/bin/python tools/test.py configs/france/dinov3-convnext-b_upernet_1xb2-amp_france-1024x1024.py \
    checkpoints/convnext_b_upernet_france_ep43.pth
PYTHONPATH=. ~/py311/bin/python tools/test.py configs/france/dinov3-vitl16-adapter_upernet_1xb2-amp_france-1024x1024.py \
    checkpoints/vitl16_adapter_upernet_france.pth
```
Оценка выполняется, как в отчётах: валидационные фрагменты целиком, скользящее окно 1024 × 1024 с шагом 768, fp32. Модель `StreamSlideEncoderDecoder` накапливает предсказания окон полосами, а не для всего фрагмента сразу, поэтому фрагмент в 120 млн пикселей требует не более 5 ГБ видеопамяти. При штатном способе одни логиты целого фрагмента заняли бы 6,9 ГБ, и ещё столько же на каждое окно. Результат совпадает со штатным скользящим окном MMSegmentation пиксель в пиксель. На RTX 3090 оценка четырёх фрагментов занимает около 30 с у SegFormer-B0, около 2 минут у ConvNeXt-B + UPerNet и около 4 минут у ViT-L + UPerNet (до 8 ГБ видеопамяти в fp32).

## Обучение
```bash
PYTHONPATH=. ~/py311/bin/python tools/train.py configs/france/segformer_mit-b0_1xb8-amp_france-1024x1024.py \
    --work-dir work_dirs/segformer_b0_france
PYTHONPATH=. ~/py311/bin/python scripts/prepare_dinov3_weights.py   # начальные веса DINOv3 ConvNeXt-B -> checkpoints/
PYTHONPATH=. ~/py311/bin/python tools/train.py configs/france/dinov3-convnext-b_upernet_1xb2-amp_france-1024x1024.py \
    --work-dir work_dirs/convnext_b_upernet_france
PYTHONPATH=. ~/py311/bin/python tools/train.py configs/france/dinov3-vitl16-adapter_upernet_1xb2-amp_france-1024x1024.py \
    --work-dir work_dirs/vitl16_adapter_upernet_france
```
Начальные веса: у SegFormer-B0 — кодировщик `mit_b0_loveda_encoder.pth` из релиза (`python3 scripts/fetch.py init`), у ConvNeXt-B + UPerNet — DINOv3 ConvNeXt-B (LVD-1689M), которые `scripts/prepare_dinov3_weights.py` скачивает с Hugging Face (репозиторий timm, с проверкой SHA-256) и переводит в формат официального кода DINOv3 тем же преобразованием, что в эксперименте. У ViT-L + UPerNet — официальный файл Meta `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (DINOv3 ViT-L/16, LVD-1689M). Его выдаёт Meta после принятия DINOv3 License (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), файл кладётся в `checkpoints/`. Конфигурация загружает его без преобразования, со строгой проверкой имён и размеров.

Параметры обучения те же, что в эксперименте:
* SegFormer-B0 — параметры, предложенные авторами SegFormer для снимков 1024 × 1024: AdamW, скорость обучения 6·10⁻⁵ (у декодера в 10 раз выше), батч 8, проверка каждые 2 «эпохи» эксперимента (эпоха — 513 итераций). На RTX 3090 итерация занимает около 0,4 с.
* ConvNeXt-B + UPerNet — параметры MMSegmentation для UPerNet с кодировщиком ConvNeXt: AdamW, скорость обучения 1·10⁻⁴ с уменьшением от поздних стадий к ранним (множитель 0,9), затухание весов 0,05, стохастическое выключение блоков 0,4, вспомогательный декодер FCN, батч 2, формат channels_last, проверка после каждой эпохи (2049 итераций) в fp16. На RTX 3090 итерация занимает около 0,44 с, обучение в эксперименте заняло 14,5 ч.
* ViT-L + UPerNet — первые 12 блоков трансформера заморожены, обучаются последние 12 блоков, адаптер и декодеры (204,6 млн параметров). AdamW, скорость обучения 1·10⁻⁴ у адаптера и декодеров, у блоков трансформера в 5 раз меньше с уменьшением к ранним блокам (множитель 0,9, правило ViT-Adapter-L), затухание весов 0,05, вспомогательный декодер FCN, батч 2, проверка после каждой эпохи (2049 итераций) в fp16. На RTX 3090 итерация занимает около 0,9 с и около 17 ГБ видеопамяти. В эксперименте лучшая эпоха — 19-я, обучение заняло около 13 ч.

У всех моделей разогрев 1500 итераций, смешанная точность fp16 и одинаковые аугментации: масштаб 0,75–1,5, вырезка 1024 × 1024, мягкие изменения яркости, контраста, насыщенности и тона, повороты на 90° и отражения. Скорость обучения снижается вдвое после 5 проверок без роста mIoU, обучение останавливается после 12 таких проверок, лучшая модель сохраняется как `best_mIoU_iter_N.pth`.

Отличие от эксперимента одно: патчи берутся штатно, случайными вырезками из плиток 2048 × 2048, тогда как в эксперименте каждая эпоха покрывала сетку обучающих фрагментов целиком и добавляла 25 % патчей на редких классах. Поэтому повторное обучение даст результат того же уровня, но не совпадающий до сотых. Дообучение ViT-L + UPerNet на дополнительных фрагментах в репозиторий не входит: эти фрагменты не опубликованы в наборе.

## Состав
* `_base_/datasets/france_cosia.py` — набор France в структуре `prepare_data.py`, 14 классов и палитра CoSIA, конвейеры обучения и оценки, метрика.
* `_base_/default_runtime.py` — рантайм и подключение модулей проекта (`custom_imports`).
* Модули проекта (`france_mmseg/`), которых нет в MMSegmentation: `StreamSlideEncoderDecoder` (потоковый скользящий инференс), `RandomDihedral` (повороты на 90° и отражения), `PlateauControlHook` (снижение скорости обучения и ранняя остановка по mIoU), `SpeedOptionsHook` (формат channels_last), замена механизма внимания MiT на `scaled_dot_product_attention` с тем же результатом и меньшим расходом памяти, `DINOv3ConvNeXt` (`dinov3_backbones.py`, обёртка официального кода DINOv3, подключается только конфигурацией ConvNeXt-B), а также `DINOv3ViTAdapter` и `ViTTopLayerDecayConstructor` (`dinov3_vit_adapter.py`: официальные ViT-L/16 и адаптер DINOv3, частичная разморозка блоков, группы скорости обучения; там же прослойки совместимости кода DINOv3 с PyTorch 2.1 и MMCV 2.1; подключаются только конфигурацией ViT-L).

Лицензии: веса ConvNeXt-B + UPerNet и ViT-L + UPerNet получены из DINOv3 и распространяются на условиях DINOv3 License (`LICENSE_DINOv3.md`). Код DINOv3 скачивается из репозитория facebookresearch/dinov3 и в этот репозиторий не входит.
