# The density of cross-persistence diagrams and its applications

Код к статье IEEE:
**A. Mironenko, E. Burnaev, S. Barannikov — "The density of cross-persistence diagrams and its applications"**  
IEEE Xplore: https://ieeexplore.ieee.org/document/11417786

Этот репозиторий подготовлен в формате "краткой публикационной версии":
- объединены рабочие версии из веток `master` и `ZH_exps`,
- удалены побочные/черновые эксперименты, не относящиеся к статье,
- у ноутбуков очищены outputs и execution counters.

## Структура репозитория

- `utils.py`  
  Общие утилиты: ragged-слои для Cross-RipsNet, вычисление расстояний и метрик между распределениями.

- `persistence density estimation.ipynb`  
  Основные эксперименты по плотности MTD (MNIST, CIFAR10, COIL20, CIFAR100, текстовые облака) — соответствует разделам про density-based distinction.

- `anti_noise_exp_1.ipynb`  
  Эксперименты с шумом для усиления различимости облаков — соответствует noise-части статьи.

- `gravity_MTD.ipynb`  
  Time-series кейс (gravitational waves) из прикладной части.

- `Cross_RipsNet.ipynb`  
  Базовый пайплайн Cross-RipsNet на синтетических облаках.

- `Cross_RipsNet_3d.ipynb`  
  Cross-RipsNet для 3D point-cloud данных (ModelNet-подобный сценарий).

- `Cross_RipsNet_text.ipynb`  
  Cross-RipsNet на текстовом кейсе (human vs GPT).

- `Cross_RipsNet_encode_exp.ipynb`  
  Эксперименты с признаками матрицы расстояний (PCA / MAX / QUANT) и сравнение вариантов архитектуры.

## Зависимости

Минимум нужны:
- Python 3.10+,
- Jupyter,
- `numpy scipy pandas matplotlib seaborn scikit-learn tqdm`,
- `tensorflow`, `torch`, `torchvision`,
- `gudhi`, `giotto-tda`, `POT`, `statsmodels`, `transformers`, `trimesh`.

Также нужны внешние проекты, которые используются в ноутбуках:
1. https://github.com/IlyaTrofimov/MTopDiv
2. https://github.com/primozskraba/PersistenceUniversality
3. https://github.com/giotto-ai/giotto-tda

## Быстрый запуск

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter numpy scipy pandas matplotlib seaborn scikit-learn tqdm tensorflow torch torchvision gudhi giotto-tda POT statsmodels transformers trimesh plotly pillow
```

После этого:
1. Положите данные в директории, которые ожидают ноутбуки (`Data/`, `RipsNet_exp/`, `COIL-20_data/`, и т.д.).
2. Убедитесь, что `MTopDiv` и `PersistenceUniversality` доступны в `PYTHONPATH` (или лежат рядом с репозиторием).
3. Запускайте ноутбуки из корня репозитория, чтобы относительные пути к данным совпадали.

## Рекомендуемый порядок воспроизведения

1. `persistence density estimation.ipynb`
2. `anti_noise_exp_1.ipynb`
3. `Cross_RipsNet.ipynb`
4. `Cross_RipsNet_3d.ipynb`
5. `Cross_RipsNet_text.ipynb`
6. `Cross_RipsNet_encode_exp.ipynb`
7. `gravity_MTD.ipynb`

## Примечание

В репозитории нет больших датасетов и артефактов обучения — оставлен только код, необходимый для воспроизведения экспериментов статьи.
