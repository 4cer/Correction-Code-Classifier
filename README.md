# AO_Projekt_1
Pierwszy projekt z AO (Algorytmy Optymalizacji)

## 1. Opis problemu
Celem projektu jest stworzenie klasyfikatora rozpoznającego obecność oraz typ jednego z trzech kodów korekcyjnych w wiadomości. Wiadomość przekazywana jest w formie ciągu zer i jedynek, zaczynając od początku symulowanej transmisji.
### Wybrane algorytmy
- Kody Hamminga
- Kody BCH
- Kody RS
### Metoda klasyfikacji
Jako metodę klasyfikacji wybrano konwolucyjną sieć neuronową (CNN). Ponieważ rozważany jest zakres długości bloków każdego z kodów, a okno wejścia sieci CNN ma stałą długość, klasyfikator będzie iterował po sygnale krokiem długości 1. Założono że rozmiar okna wejściowego będzie wynosił 3-krotność największego rozmiaru bloku.

## 2. Schemat struktury projektu
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/hsHuMpx.jpg)
</details>
Opis Lorem Ipsum

## 3. Schemat tworzenia datasetu
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/XhlgDxg.jpg)
</details>
Opis Lorem Ipsum

## 4. Schemat modelu klasyfikującego
<details>
  <summary>Duża grafika</summary>

  ![GRAFIKA W PRZYGOTOWANIU](https://i.imgur.com/emoB4tF.jpg)
</details>
Opis Lorem Ipsum

### 4.1. Szczegółowy schemat sieci konwolucyjnej
<details>
  <summary>Duża grafika</summary>

  ![GRAFIKA W PRZYGOTOWANIU](http://url/to/img.png)
</details>
Opis Lorem Ipsum
