# AO_Projekt_1
Pierwszy projekt z AO (Algorytmy Optymalizacji)

## 1. Opis problemu
<p>Celem projektu jest stworzenie klasyfikatora rozpoznającego obecność oraz typ jednego z trzech kodów korekcyjnych w wiadomości.</p>
<p>Wiadomość przekazywana jest w formie ciągu zer i jedynek, zaczynając od początku symulowanej transmisji.</p>

### Wybrane algorytmy
<ol start='0'>
  <li>Brak kodu</li>
  <li>Kody Hamminga</li>
  <li>Kody BCH</li>
  <li>Kody RS</li>
</ol>

<p>Numerowanie powyższego wyliczenia odpowiada również numeracji klas w implementacji</p>

### Metoda klasyfikacji
Jako metodę klasyfikacji wybrano model hybrydowy: konwolucyjną incepcyjną sieć neuronową. Ponieważ rozważany jest zakres długości bloków każdego z kodów, a okno wejścia sieci CNN ma stałą długość, klasyfikator będzie iterował po sygnale krokiem długości 1. Założono że rozmiar okna wejściowego będzie wynosił 3-krotność największego rozmiaru bloku.

## 1. Schemat struktury projektu
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/hsHuMpx.jpg)
</details>
<p>Każdy zasadniczy moduł projektu posiada własną podścieżkę w folderze głównym projektu. Moduły wyróżniono według funkcjonalności:</p>
<ul>
  <li>./packages_datasetgen - tworzenie zbioru danych</li>
  <li>./packages_gio - wczytywanie i preparacja danych do zastosowania</li>
  <li>./packages_classifier - model klasyfikujący i jego zależności</li>
</ul>

## 2. Schemat tworzenia datasetu
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/XhlgDxg.jpg)
</details>
<p>Zbiór danych powstaje na podstawie losowego szumu, który następnie zostaje zakodowany losowo wybranym algorytmem i wraz z opisem zapisany.</p>
<p>Badania przeprowadzone w ramach projektu pokazują że losowy szum cechuje niski współczynnik informacji do szumu (Signal-to-noise ration, SNR), przez co trening modelu przynosi znikome efekty dla optymalizatora adam.</p>

## 3. Schemat modelu klasyfikującego
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/BXeJ7nn.jpg)
</details>
<p>Przedstawiony model hybrydowy w pierwszych warstwach przebiega podobnie do zwykłego modelu CNN, jednak po pierwszej parze konwolucji oraz poolingu dokonywane są 3 przejścia przez warstwy incepcyjne.</p>
<p>Po każdej warstwie konwolucyjnej (włączając te wewnątrz warstwy incepcyjnej) zastosowano funkcję ReLU, aby zapobiec zjawisku zanikającego gradientu oraz wprowadzić nieliniowość.</p>
<p>Ostatnimi warstwami sieci są dwie warstwy liniowe. Aby odzwierciedlić propabilistyczną naturę wyniku klasyfikacji, wektor odpowiedzi sieci traktowany jest jako prawdopodobieństwa przynależności danych do każdej z 4 klas; funckja softmax zapewnia sumowanie wartości do 1.</p>

<!-- ### 4.1. Szczegółowy schemat sieci konwolucyjnej
<details>
  <summary>Duża grafika</summary>

  ![GRAFIKA W PRZYGOTOWANIU](http://url/to/img.png)
</details>
Opis Lorem Ipsum -->

## 4. Wyniki działania
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/rl8QAel.png)
</details>
<p>Niestety dla utworzonego zbioru danych wynik jest nieznacznie gorszy od losowego zgadywania - dla 4 możliwości teoretyczna szansa trafienia wynosi 25%; klasyfikator dobrze trafia z prawdopodobieństwem 0.244</p>

## Wnioski
Zbiór danych utworzony na podstawie szumu, mimo wprowadzenia źródła informacji jakim jest kodowanie ciągów, nie nadaje się do treningu sieci. Podstawę zbioru danych musi stanowić źródło sygnałów o wysokim wskaźniku SNR, gdyż w przeciwnym wypadku sieci neuronowe nie wykazują postępów w treningu.
