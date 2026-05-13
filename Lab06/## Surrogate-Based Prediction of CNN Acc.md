## Surrogate-Based Prediction of CNN Accuracy for CIFAR-10

Należy zaimplementować system przewidujący wartość `Accuracy` architektury CNN dla zbioru CIFAR-10 wyłącznie na podstawie opisu architektury sieci neuronowej.

Projekt powinien wykorzystywać:

* lokalny model LLM generujący architektury CNN,
* modele surogatowe:

  * MLP,
  * XGBoost XGBoost.

---

# Wymagania projektowe

## 1. Dataset

Należy wykorzystać zbiór danych:

* CIFAR-10

---

# 2. Generator architektur

Architektury CNN mają być generowane przez lokalnie uruchomiony model LLM.

Przykładowe rozwiązania:

* [Ollama](https://ollama.com

LLM powinien generować architektury w formacie JSON.


Przykład:

```json id="9jghzp"
{
  "layers": [
    {"type":"conv","filters":32,"kernel":3},
    {"type":"conv","filters":64,"kernel":3},
    {"type":"maxpool"},
    {"type":"dropout","p":0.2},
    {"type":"linear","units":128}
  ]
}
```

---

# 3. Search Space

Dozwolone typy warstw:

```text id="ms33ak"
Conv2D
BatchNorm
ReLU
MaxPool
Dropout
Linear
GlobalAveragePooling
```

Ograniczenia:

* maksymalnie 6 warstw konwolucyjnych,
* poprawność wymiarowa architektury,
* ograniczenie liczby parametrów.

---

# 4. Początkowy dataset architektur

Na początku należy wygenerować początkowy zbiór architektur CNN.

Minimalna liczba:

```text id="dkph0l"
50–100 architektur
```

Każda architektura powinna zostać:

1. zbudowana w PyTorch lub TensorFlow,
2. wytrenowana na CIFAR-10,
3. oceniona metryką Accuracy.

Dla każdej architektury należy zapisać:

| Architektura | Accuracy |
| ------------ | -------- |
| CNN #1       | 78.4%    |
| CNN #2       | 81.1%    |

---

# 5. Encoding architektury

Architektura CNN musi zostać zamieniona na reprezentację numeryczną.

Modele:

* MLP,
* XGBoost

nie mogą przetwarzać JSON bezpośrednio.

Należy przygotować fixed-length vector opisujący architekturę.

---

# Przykład

## Architektura

```text id="fdff44"
Conv(32,3)
Conv(64,3)
MaxPool
Dropout(0.2)
Linear(128)
```

---

## Reprezentacja numeryczna

```text id="8mvx55"
[
  1,32,3,
  1,64,3,
  2,0,0,
  3,0.2,0,
  4,128,0
]
```

gdzie:

| Kod | Warstwa |
| --- | ------- |
| 1   | Conv    |
| 2   | MaxPool |
| 3   | Dropout |
| 4   | Linear  |

---

# 6. Modele surogatowe

Należy zaimplementować:

## a) MLP

Sieć neuronową przewidującą Accuracy architektury.

---

## b) XGBoost

Model boostingowy dla danych tabelarycznych.

---

# 7. Zadanie modeli surogatowych

Modele mają realizować predykcję:

[
f(\text{architektura}) \rightarrow \text{Accuracy}
]

czyli:

```text id="u5sox4"
wektor architektury → przewidywane Accuracy
```

---

# 8. Iteracyjny pipeline uczenia

Po nauczeniu pierwszego surrogate model należy uruchomić iteracyjny proces eksploracji architektur.

---

# Iteracja

## Krok 1

LLM generuje nową architekturę CNN.

---

## Krok 2

Architektura zostaje zakodowana do wektora liczbowego.

---

## Krok 3

Surrogate model przewiduje Accuracy.

Przykład:

```text id="l5t80t"
Predicted Accuracy = 84.2%
```

---

## Krok 4

Jeżeli surrogate model uzna, że architektura jest lepsza od wcześniejszych:

```text id="t7i2n5"
predicted accuracy > current best accuracy
```

to architektura:

1. jest trenowana w pełni,
2. obliczane jest rzeczywiste Accuracy.

---

## Krok 5

Nowa architektura wraz z rzeczywistym Accuracy zostaje dodana do datasetu.

---

## Krok 6

Surrogate model jest uczony ponownie na rozszerzonym datasetcie.

---

# 9. Metryki ewaluacji

Należy wykorzystać:

---

## MAE — Mean Absolute Error

Python:

```python id="n3r44w"
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

---

## RMSE — Root Mean Squared Error


```python id="kmj7y5"
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

---

## (R^2) — Coefficient of Determination


```python id="b57drx"
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

---

## Spearman Correlation

Metryka zgodności rankingu architektur.

Python:

```python id="c5y9wp"
from scipy.stats import spearmanr

corr, _ = spearmanr(y_true, y_pred)
```

---

