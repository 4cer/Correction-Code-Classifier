# AO_Projekt_1
Pierwszy projekt z AO (Algorytmy Optymalizacji)

## 1. Opis problemu
Celem projektu jest stworzenie klasyfikatora rozpoznającego obecność oraz typ jednego z trzech kodów korekcyjnych w wiadomości. Wiadomość przekazywana jest w formie ciągu zer i jedynek, zaczynając od początku symulowanej transmisji.
### Wybrane algorytmy
<ol start='0'>
  <li>Brak kodu</li>
  <li>Kody Hamminga</li>
  <li>Kody BCH</li>
  <li>Kody RS</li>
</ol>

Numerowanie powyższego wyliczenia odpowiada również numeracji klas w implementacji

### Metoda klasyfikacji
Jako metodę klasyfikacji wybrano model hybrydowy: konwolucyjną incepcyjną sieć neuronową. Ponieważ rozważany jest zakres długości bloków każdego z kodów, a okno wejścia sieci CNN ma stałą długość, klasyfikator będzie iterował po sygnale krokiem długości 1. Założono że rozmiar okna wejściowego będzie wynosił 3-krotność największego rozmiaru bloku.

## 1. Schemat struktury projektu
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/hsHuMpx.jpg)
</details>
Każdy zasadniczy moduł projektu posiada własną podścieżkę w folderze głównym projektu. Moduły wyróżniono według funkcjonalności:
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
Zbiór danych powstaje na podstawie losowego szumu, który następnie zostaje zakodowany losowo wybranym algorytmem i wraz z opisem zapisany.<br>
Badania przeprowadzone w ramach projektu pokazują że losowy szum cechuje niski współczynnik informacji do szumu (Signal-to-noise ration, SNR), przez co trening modelu przynosi znikome efekty dla optymalizatora adam.

## 3. Schemat modelu klasyfikującego
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/BXeJ7nn.jpg)
</details>
Przedstawiony model hybrydowy w pierwszych warstwach przebiega podobnie do zwykłego modelu CNN, jednak po pierwszej parze konwolucji oraz poolingu dokonywane są 3 przejścia przez warstwy incepcyjne.<br>
Po każdej warstwie konwolucyjnej (włączając te wewnątrz warstwy incepcyjnej) zastosowano funkcję ReLU, aby zapobiec zjawisku zanikającego gradientu oraz wprowadzić nieliniowość.<br>
Ostatnimi warstwami sieci są dwie warstwy liniowe. Aby odzwierciedlić propabilistyczną naturę wyniku klasyfikacji, wektor odpowiedzi sieci traktowany jest jako prawdopodobieństwa przynależności danych do każdej z 4 klas; funckja softmax zapewnia sumowanie wartości do 1.

<!-- ### 4.1. Szczegółowy schemat sieci konwolucyjnej
<details>
  <summary>Duża grafika</summary>

  ![GRAFIKA W PRZYGOTOWANIU](http://url/to/img.png)
</details>
Opis Lorem Ipsum -->
