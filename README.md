# AO_Projekt_1
Pierwszy projekt z AO (Algorytmy Optymalizacji)

## 1. Opis problemu
<p>Celem projektu jest stworzenie klasyfikatora rozpoznającego obecność oraz typ jednego z trzech kodów korekcyjnych w wiadomości składającej się z niezaszyfrowanego języka natualnego. W hipotetycznej sytuacji istnieje podejrzenie zastosowania algorytmu kodowania korekcyjnego; celem modelu jest więc potwierdzenie podejrzenia oraz przybliżenie rodziny algorytmów.</p>
<p>Wiadomość przekazywana jest w formie ciągu zer i jedynek, przy czym zakłada się że ciąg został przechwycony od początku lub wykorzystano inne techniki by znaleźć pierwszy pełny blok informacji i odrzucono poprzedzające go informacje.</p>

### Wybrane algorytmy
<ol start='0'>
  <li>Brak kodu</li>
  <li>Kody Hamminga</li>
  <li>Kody BCH</li>
  <li>Kody RS</li>
</ol>

<p>Numerowanie powyższego wyliczenia odpowiada również numeracji klas w implementacji</p>

### Metoda klasyfikacji
Jako metodę klasyfikacji wybrano model hybrydowy: konwolucyjną incepcyjną sieć neuronową. Ponieważ rozważany jest zakres długości bloków każdego z kodów, a okno wejścia sieci CNN ma stałą długość, przyjmuje się że klasyfikator otrzymuje na wejście spreparowane wycinki sygnału o określonej długości - 3072 bitów. W hipotetycznym scenariuszu zastosowania, klasyfikator mógłby być wykorzystywany do ciągłej iteracji po nadchodzącym sygnale, jednak nie jest to kierunek rozważany w trakcie szkolenia sieci.

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

  ![alt text](https://i.imgur.com/o6ZAdNb.jpg)
</details>

### 2.1. Podejście pierwsze
<p>Pierwotnie generację opracowano tak, by zbiór danych powstawał poprzez generowanie losowych ciągów bitów i kodowanie losowym algorytmem. Liczba punktów danych wyniosła 10,000 ciągów o długości 384 bitów. W trakcie szkolenia przez 500 epok sieć nie wykazała poprawy celności klasyfikacji na zbiorze treningowym ani testowym.<p>
<p>Przetestowano szerego potencjalnych rozwiązań, w tym: zwiększenie długości zakodowanych ciągów do 3072 bitów (co reprezentuje 8-krotne zwiększenie ilości informacji dla każdego punktu danych); Dodanie do danych kontekstu poprzez zwiększenie liczby próbek do 40,000 i kodowanie każdego z ciągów każdym z algorytmów. Żadne z rozwiązań nie prowazdiło</p>
<p>Badania przeprowadzone w ramach projektu pokazują że dane tworzone z wykorzystaniem losowego szumu cechuje wysoka entropia powodująca efekt zbliżony do niskiego współczynnika informacji do szumu (Signal-to-noise ration, SNR), przez co trening modelu przynosi znikome efekty dla optymalizatora adam. Konieczna jest zmiana źródła informacji wejściowych.</p>

### 2.2. Podejście zrewidowane
<p>Zbiór danych powstaje na podstawie próbkowania treści lektur szkolnych:</p>
<details>
  <summary>Lista</summary>
  <ul>
    <li>arystoteles - poetyka</li>
    <li>ave maria</li>
    <li>cassanova de seingalt - od kobiety do kobiety</li>
    <li>don kichot z la manchy</li>
    <li>dzieje tristana i izoldy</li>
    <li>giaur</li>
    <li>historia zoltej cizemki</li>
    <li>hoffmann - dziadek do orzechow</li>
    <li>homer - odyseja</li>
    <li>kariera nikodema dyzmy</li>
    <li>pan tadeusz</li>
    <li>przygody tomka sawyera</li>
    <li>quo vadis</li>
    <li>robinson crusoe</li>
    <li>rozprawa o metodzie</li>
    <li>spinoza - etyka</li>
    <li>syzyfowe prace</li>
    <li>tajemniczy ogrod</li>
    <li>w pustyni i w puszczy</li>
    <li>znachor</li>
  </ul>
</details>
<p>Lektury wczytywane są jako ciągi znaków w kodowaniu UTF-8 i przechowywane w tabeli, po wstępnej preparacji: każda z zawartości zostaje znormalizowana poprzez usunięcie przedmów, komentarzy oraz innych informacji niebędących treścią; następnie usuwane są wielokrotne powtórzenia znaku spacji, oraz wszystkie znaki nowej linii.</p>
<p>Iteracje tworzące ciągi bitów z losowo wybranej lektury próbkują 386 znaków z losowym przesunięciem, które zostają przedstawione jako ciąg zer i jedynek po czym zakodowane każdym z algorytmów kodowania korekcyjnego. Dla 40,000 punktów danych utworzonych w ten sposób otrzymano trafność klasyfikacji na poziomie 76%. Dalsze eksperymenty z parametrami modelu oraz algorytmu optymalizującego osiągnięto trafność 1.0. Ponieważ mniejszy zestaw danych posiada wyłącznie po jednym koderze każdego rodzaju, model trenowany na tej wersji nie potrafi generalizować.</p>
<p>Ostatnim krokiem było utworzenie wersji zbioru o rozmiarze 400,000 próbek i po 3 kodery na algorytm z różnymi parametrami. Z początku powrócił problem zerowych postępów treningu, więc wprowadzono wykładniczą zmienność tempa nauki co przyniosło wymierne efekty.</p>

## 3. Schemat modelu klasyfikującego
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/70Hlvc0.jpg)
</details>
<p>Przedstawiony model hybrydowy w pierwszych warstwach przebiega podobnie do zwykłego modelu CNN, jednak po pierwszej parze konwolucji oraz poolingu dokonywane są 3 przejścia przez warstwy incepcyjne.</p>
<p>Po każdej warstwie konwolucyjnej (włączając te wewnątrz warstwy incepcyjnej) zastosowano funkcję ReLU, aby zapobiec zjawisku zanikającego gradientu oraz wprowadzić nieliniowość.</p>
<p>Ostatnimi warstwami sieci są dwie warstwy liniowe. Aby odzwierciedlić propabilistyczną naturę wyniku klasyfikacji, wektor odpowiedzi sieci traktowany jest jako prawdopodobieństwa przynależności danych do każdej z 4 klas; funckja softmax zapewnia sumowanie wartości do 1.</p>

## 4. Wyniki działania
### 4.1. Zbiór danych oparty na zakodowanym szumie
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/rl8QAel.png)
</details>
<p>Niestety dla utworzonego zbioru danych wynik jest nieznacznie gorszy od losowego zgadywania - dla 4 możliwości teoretyczna szansa trafienia wynosi 25%; klasyfikator dobrze trafia z prawdopodobieństwem 0.244</p>

### 4.2. Zbiór danych oparty na próbkowaniu lektur
#### 4.2.1. Zbiór mniejszy
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/n0bGeXb.png)
</details>
<p>Wynik dla mniejszej wersji drugiego zbioru danych (40,000 próbek) wynosi 76%, co stanowi ponad trzykrotną poprawę: 3 na 4 predykcje są poprawne.</p>

#### 4.2.2. Zbiór większy
<details>
  <summary>Duża grafika</summary>

  ![alt text](https://i.imgur.com/rYfOcZj.png)
</details>
<p>Większa wersja zbioru (400,000 próbek) pozwoliła osiągnąć prawdopodobieństwo prawidłowego wyniku na poziomie 96.7%</p>
<p>Na podstawie przedstawionej macierzy błędów (<i>ang. confusion matrix</i>) widać iż najwięcej błędów (317/10,000 - 97.2%) wynikało z błędnej klasyfikacji kodowania próbek algorytmem RS jako kod Hamminga, drugie miejsce (7/10,000 - 2.1%) to niezakodowane próbki rozpoznane jako przkłady kodu RS.</p>

## 5. Metodyka treningu
<p>Model we wczesnych etapach trenowano lokalnie, jednak największy wariant zbioru danych wymaga większej ilości pamięci podręcznej RAM nawet by wczytać dane. Próby wykonywane na przestrzeni rozwoju wykazały również potrzebę wykorzystania procesorów graficznych (GPU) w procesie optymalizacji. Na podstawie przeszłych doświadczeń wybrano technologię CUDA wykorzystywaną przez karty graficzne firmy nvidia.</p>
<p>Początkowo wykorzystywano posiadaną kartę graficzna z serii 1600, która okazała się niedostateczna dla dużego zbioru danych. W związku z tym problemem wykorzystano platformę Google Colab, na której wykonano ostateczny trening przez 115 na 200 zaplanowanych epok trwający 4h 23min. Warto zanotować iż na ostatnią poprawę zanotowano w okolicy 30 epoki, w związku z czym można dokonać spekulacji iż 200 epok nie było konieczne.</p>

## Wnioski
<p>Zbiór danych utworzony na podstawie szumu, mimo wprowadzenia źródła informacji jakim jest kodowanie ciągów, nie nadaje się do treningu sieci. Podstawę zbioru danych musi stanowić źródło sygnałów o niskiej entropii, gdyż w przeciwnym wypadku sieci neuronowe nie wykazują postępów w treningu.</p>
<p>W celu osiągnięcia zadowalających efektów wykorzystano zbiór danych zawierający 400,000 próbek oraz platformę Google Colab pozwalającą na bezpłatne przeprowadzenie treningu na procesorze graficznym Nvidia Tesla T4.</p>
<p>Próby wykorzystania modelu trenowanego na mniejszym i mniej różnorodnym zbiorze danych osiągają znacząco niższą skuteczność (przykładowo 51.1%) przy zastosowaniu na większym i bardziej różnorodnym zbiorze danych.</p>
<br>
<p>todo dodać wnioski</p>