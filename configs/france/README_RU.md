# Конфигурация MMSegmentation для SegFormer-B0 на наборе France

Конфигурация в формате MMSegmentation 1.2.2 (наследование от `_base_`, штатные `BaseSegDataset`, `IoUMetric`, преобразования MMSegmentation) запускается штатными `tools/train.py` и `tools/test.py`.

| Конфигурация | Модель | Веса | mIoU (val, 14 классов) |
|---|---|---|---|
| `segformer_mit-b0_1xb8-amp_france-1024x1024.py` | SegFormer-B0 | `segformer_b0_france_ep76.pth` ([релиз](https://github.com/tester170/france-segformer-b0/releases/tag/v1.0)) | 72,90 |

Значение mIoU получено штатным `tools/test.py` с этой конфигурацией и весами и совпадает с отчётом.

## Подготовка
```bash
bash env/setup_env.sh ~/py311                         # Python 3.11, PyTorch 2.1.0, MMCV 2.1.0, MMSegmentation 1.2.2
python3 scripts/fetch.py dataset --out data/France    # набор France с Яндекс Диска, проверка SHA-256
~/py311/bin/python scripts/prepare_data.py --src data/France --out data/france
python3 scripts/fetch.py model                        # обученная модель -> checkpoints/
python3 scripts/fetch.py init                         # кодировщик LoveDA -> checkpoints/ (для обучения)
```
`prepare_data.py` раскладывает набор в структуру MMSegmentation: `data/france/img_dir|ann_dir/train` — плитки 2048 × 2048 двенадцати обучающих фрагментов, `img_dir|ann_dir/val` — четыре валидационных фрагмента целиком. Метки переводятся из кодов CoSIA (0–5, 7–14) в номера классов 0–13. Команды ниже запускаются из корня репозитория с `PYTHONPATH=.`: так MMSegmentation находит модули `france_mmseg`, перечисленные в `custom_imports`.

## Оценка
```bash
PYTHONPATH=. ~/py311/bin/python tools/test.py configs/france/segformer_mit-b0_1xb8-amp_france-1024x1024.py \
    checkpoints/segformer_b0_france_ep76.pth
```
Оценка выполняется, как в отчёте: валидационные фрагменты целиком, скользящее окно 1024 × 1024 с шагом 768, fp32. Модель `StreamSlideEncoderDecoder` накапливает предсказания окон полосами, а не для всего фрагмента сразу, поэтому фрагмент в 120 млн пикселей требует около 4,5 ГБ видеопамяти. При штатном способе одни логиты целого фрагмента заняли бы 6,9 ГБ, и ещё столько же на каждое окно. Результат совпадает со штатным скользящим окном MMSegmentation пиксель в пиксель.

## Обучение
```bash
PYTHONPATH=. ~/py311/bin/python tools/train.py configs/france/segformer_mit-b0_1xb8-amp_france-1024x1024.py \
    --work-dir work_dirs/segformer_b0_france
```
Параметры обучения те же, что в эксперименте, и соответствуют параметрам, предложенным авторами SegFormer для снимков 1024 × 1024: AdamW, скорость обучения 6·10⁻⁵ (у декодера в 10 раз выше), разогрев 1500 итераций, батч 8, смешанная точность fp16. Аугментации: масштаб 0,75–1,5, вырезка 1024 × 1024, мягкие изменения яркости, контраста, насыщенности и тона, повороты на 90° и отражения. Проверка выполняется каждые 2 «эпохи» эксперимента (513 итераций). Скорость обучения снижается вдвое после 5 проверок без роста mIoU, обучение останавливается после 12 таких проверок, лучшая модель сохраняется как `best_mIoU_iter_N.pth`. На RTX 3090 итерация занимает около 0,4 с.

Отличие от эксперимента одно: патчи берутся штатно, случайными вырезками из плиток 2048 × 2048, тогда как в эксперименте каждая эпоха покрывала сетку обучающих фрагментов целиком и добавляла 25 % патчей на редких классах. Поэтому повторное обучение даст результат того же уровня, но не совпадающий до сотых.

## Состав
* `_base_/datasets/france_cosia.py` — набор France в структуре `prepare_data.py`, 14 классов и палитра CoSIA, конвейеры обучения и оценки, метрика.
* `_base_/default_runtime.py` — рантайм и подключение модулей проекта (`custom_imports`).
* Модули проекта (`france_mmseg/`), которых нет в MMSegmentation: `StreamSlideEncoderDecoder` (потоковый скользящий инференс), `RandomDihedral` (повороты на 90° и отражения), `PlateauControlHook` (снижение скорости обучения и ранняя остановка по mIoU) и замена механизма внимания MiT на `scaled_dot_product_attention` с тем же результатом и меньшим расходом памяти.
