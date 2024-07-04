# Wykrywacz marek pojazdow z rejestracja

<!-- Badges -->
<p>
  <a href="https://github.com/kasperserzysko/carClasificatorgraphs/contributors">
    <img src="https://img.shields.io/github/contributors/kasperserzysko/carClasificator" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/kasperserzysko/carClasificator" alt="last update" />
  </a>
</p>

<!-- Table of Contents -->


<!-- About the Project -->
## 🤩 O projekcie
Projekt ma na celu wprowadzenie szczegolowego systemu detekcji. Bedzie on zajmowal sie rozpoznawaniem marek samochodow
oraz ich tablic rejestracyjnych. Poczatkowe zalozenie jest takie iz do rozpoznawanie bedzie mozliwe po zdjeciach. Natomiast po
uzyskaniu zadowalojacych wynikow bedziemy mogli przeniesc modele na rozponzwanie w czasie rzeczywistym z nagran wideo.




## Zbiory danych
[Zbior danych tablic rejestracyjnych](https://huggingface.co/datasets/keremberke/license-plate-object-detection)<br>
[Zbior danych blisko 20 marek samochodow](https://www.kaggle.com/datasets/alirezaatashnejad/over-20-car-brands-dataset)


# **Korzystanie ze zbiorow**
## Wymagana instalacja
```bash
poetry install
```
## Tworzenie zbiorow danych
Z powodow prawnych nie mozemy publikowac uzytych zbiorow danych. Natomiast jest mozliwosc pobraniach ich idywidualnie i uruchomeina skryptow do ich przeprocesowania.

<details>
  <summary>1. Wykrywanie tablic rejestracyjnych</summary>

**5. Wykrywanie tablic rejestracyjnych**
  1. Przejdz na [zbior tablic rejestracyjnych](https://huggingface.co/datasets/keremberke/license-plate-object-detection) strona HugginFace.
  2. Zaloguj sie na swoje konto.
  3. Pobierz zbior danych do glownego folderu .
  4. Wypakuj utworzony plik z rzoszerzeniem `.zip`.

</details>

<details>
  <summary>2. Wykrywanie marek pojazdow</summary>

**2.Wykrywanie marek pojazdow **
  1. Przejdz na [zbior danych marek samochodwo](https://www.kaggle.com/datasets/alirezaatashnejad/over-20-car-brands-dataset) strona Kaggle .
  2. Zaloguj sie na swoje konto Kaggle.
  3. Pobierz zbior do folderu glownego.
  4. Wypakuj `over-20-car-brands-dataset.zip` uzyj do tego pliku car-clasificator/datasets/car_brands_dataset.py.

</details>


To preprocess the dataset that is not among the above, search the preprocessing folder. It contains the reusable steps for changing imaging formats, extracting masks, creating file trees, etc. Go to the config file to check which masks and label encodings are available. Append new labels and mask encodings if needed.



## 🎯 Roadmap
- [x] Zbior danych Huggingface 
- [x] Zbior danych Kaggle 
- [ ] Zobrazowanie danych


<!-- Contributing -->
## :wave: Contributors

<a href="https://github.com/kasperserzysko/carClasificatorgraphs/contributors">
  <img src="https://contrib.rocks/image?repo=kasperserzysko/carClasificatorgraphs" />
</a>


# Development
## Pre-commits
Zainstaluj precommity
https://pre-commit.com/#installation

Jezeli uzywasz VS-code zainstalun rozszerzenie https://marketplace.visualstudio.com/items?itemName=MarkLarah.pre-commit-vscode

Uruchom precommity zeby sparwdzic czy caly kod sie aktywuje:
```
pre-commit run --all-files
```


## Dodawanie paczek
Zaleznosci sa obslugiwane przez `poetry` framework, zeby dodac nowa zaleznosc uruchom:
```
poetry add <package_name>
```

