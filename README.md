# UMIE_datasets

<!-- Badges -->
<p>
  <a href="https://github.com/kasperserzysko/carClasificatorgraphs/contributors">
    <img src="https://img.shields.io/github/contributors/kasperserzysko/carClasificator" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/kasperserzysko/carClasificator" alt="last update" />
  </a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="license" />
  </a>

</p>

<!-- Table of Contents -->


<!-- About the Project -->
## 🤩 About the Project





## Datasets
| uid | Dataset | Modality | TASK |



# **Using the datasets**
## Installing requirements
```bash
poetry install
```
## Creating the dataset
Due to the copyright restrictions of the source datasets, we can't share the files directly. To obtain the full dataset you have to download the source datasets yourself and run the preprocessing scripts.

<details>
  <summary>1. Wykrywanie tablic rejestracyjnych</summary>

**5. Wykrywanie tablic rejestracyjnych**
  1. Przejdz na [Car number plates dataset](https://huggingface.co/datasets/keremberke/license-plate-object-detection) strona HugginFace.
  2. Zaloguj sie na swoje konto.
  3. Pobierz zbior danych.
  4. Wypakuj `archive.zip`.

</details>

<details>
  <summary>2. Wykrywanie marek pojazdow</summary>

**2.Wykrywanie marek pojazdow **
  1. Przejdz na [20 car brands photos dataset](https://www.kaggle.com/datasets/alirezaatashnejad/over-20-car-brands-dataset) strona Kaggle .
  2. Zaloguj sie na swoje konto Kaggle.
  3. Pobierz zbior do folderu glownego.
  4. Wypakuj `over-20-car-brands-dataset.zip` uzyj do tego pliku car-clasificator/datasets/car_brands.py.

</details>


To preprocess the dataset that is not among the above, search the preprocessing folder. It contains the reusable steps for changing imaging formats, extracting masks, creating file trees, etc. Go to the config file to check which masks and label encodings are available. Append new labels and mask encodings if needed.

#TODO
Overall analysis of dataset here:


## 🎯 Roadmap
- [x]  Huggingface datasets
- [x] Kaggle datasets
- [ ] Data dashboards


<!-- Contributing -->
## :wave: Contributors

<a href="https://github.com/kasperserzysko/carClasificatorgraphs/contributors">
  <img src="https://contrib.rocks/image?repo=kasperserzysko/carClasificatorgraphs" />
</a>


# Development
## Pre-commits
Install pre-commits
https://pre-commit.com/#installation

If you are using VS-code install the extention https://marketplace.visualstudio.com/items?itemName=MarkLarah.pre-commit-vscode

To make a dry-run of the pre-commits to see if your code passes run
```
pre-commit run --all-files
```


## Adding python packages
Dependencies are handeled by `poetry` framework, to add new dependency run
```
poetry add <package_name>
```

## Debugging

To modify and debug the app, [development in containers](https://davidefiocco.github.io/debugging-containers-with-vs-code) can be useful .
