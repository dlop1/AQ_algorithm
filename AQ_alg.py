from bitarray.util import ones, zeros
from random import randint, seed
from copy import deepcopy

class AQ:
    """
    Attributes:
            m: int → parametr określający, ile najlepszych kompleksów pozostaje w zbiorze tworzonych kompleksów
            random_seed_choice: bool → parametr określający, czy ziarna wybierane podczas tworzenia reguły mają być wybierane w losowej kolejności, czy zgodnie z kolejnościa w zbiorze przykładów
            attributes_domains: List[List[Any]] → lista zawierająca dziedziny wszystkich atrybutów, typ danych w Set zależy od typu danych atrybutów
            data: List[Tuple[List[bitarray.bitarray], Any]] → zbiór wszystkich przykładów, przykład składa się z atrybutów oraz klasy
            non_cover_data: List[Tuple[List[bitarray.bitarray], Any]] → zbiór przykładów niepokrytych przez reguły ze zbioru `rules`
            rules: List[Tuple[List[bitarray.bitarray], Any]] → wygenerowane reguły składają się z kompleksu i klasyfikowanej klasy
            most_general_complex: List[bitarray.bitarray] → najbardziej ogólny kompleks
    """
    def __init__(self, m, random_seed_choice, domains):
        """
        Args:
            m: int → parametr określający, ile najlepszych kompleksów pozostaje w zbiorze tworzonych kompleksów
            random_seed_choice: bool → parametr określający, czy ziarna wybierane podczas tworzenia reguły mają być wybierane w losowej kolejności, czy zgodnie z kolejnościa w zbiorze przykładów
            domains: List[List[Any]] → lista zawierająca dziedziny wszystkich atrybutów, typ danych w Set zależy od typu danych atrybutów
        """
        self.m = m
        self.random_seed_choice = random_seed_choice
        self.attributes_domains = domains
        self.data = []
        self.non_cover_data = []
        self.rules = []
        self.most_general_complex = [ones(len(dom)) for dom in self.attributes_domains]
        seed(42)

    def get_data(self, data_x, data_y):
        """
        Funcja odpowiedzialna jest za początkowe przetworzenie danych wejściowych lista atrybutów
        zamieniana jest na bitarray, który zawiera 1 w miejscu występowania danej wartości
        atrybutu w dziedzinie

        Args:
            data_x: numpy.ndarray[numpy.ndarray[Any]] → lista zawierająca listy z wartościami atrybutów dla przykładów wejściowych
            data_y: numpy.ndarray[numpy.ndarray[Any]] → lista zawierająca klasy przykładów
        """
        for row, cla in zip(data_x, data_y):
            new_data = []
            for i, att in enumerate(row):
                new_data.append(self.encode_att_to_one_hot(att, i))
            self.data.append((new_data, cla[0]))
            self.non_cover_data.append((new_data, cla[0]))

    def encode_att_to_one_hot(self, attribute, i):
        """
        Funkcja konwertuje wartość i-tego atrybutu na kod 1 z n

        Args:
            attribute: Any → wartość atrybutu
            i: int → indeks atrybutu w liscie atrybutów danych

        Return:
            bitarray.bitarray → Zakodowany atrybut w kodzie 1 z n
        """
        encoded_att = zeros(len(self.attributes_domains[i]))
        encoded_att[self.attributes_domains[i].index(attribute)] = True
        return encoded_att

    def create_rules(self):
        """
        Funkcja wywołująca metodę tworzenia nowej reguły do momentu, aż zbiór non_cover_data będzie pusty
        """
        while len(self.non_cover_data) > 0:
            self.create_new_rule()

    def create_new_rule(self):
        """
        Bazujac na specjalizacji AQ algorytm tworzy nową regułę
        na bazie ziarna pozytywnego pochodzącego ze zbioru non_cover_data.
        Podczas wytwarzania reguły przykład xn wybierany jest
        z całego zbioru trenującego all_data, a nie tylko
        ze zbioru non_cover_data.
        """
        complexes = [deepcopy(self.most_general_complex)]
        xs, xs_class = self.get_new_seed()

        for x, x_class in self.data:
            if xs_class != x_class and x != xs:
                for complex in deepcopy(complexes):
                    if self.check_covering(complex, x):
                        complexes.pop(complexes.index(complex))
                        complexes = complexes + self.specialize(complex, x, xs)

                complexes = self.remove_non_maximum_general_complexes(complexes)
                complexes = self.select_best_m_complexes(self.m, complexes, xs_class)

        complexes = self.select_best_m_complexes(1, complexes, xs_class)
        self.rules.append((complexes[0], xs_class))
        self.non_cover_data.remove((xs, xs_class))

        for x, x_class in deepcopy(self.non_cover_data):
            if self.check_covering(complexes[0], x):
                self.non_cover_data.remove((x, x_class))


    def get_new_seed(self):
        """
        Funkcja zwraca ziarno pozytywne potrzebne do rozpoczęcia tworzenia nowej reguły.
        W zależności od parameru self.random_seed_choice ziarno jest wybierane w losowej kolejności lub w kolejności zgodnej z występowaniem w zbiorze danych.

        Return:
            Tuple[List[bitarray.bitarray], Any] → ziarno
        """
        index = 0
        if self.random_seed_choice:
            index = randint(0, len(self.non_cover_data) - 1)
        return (self.non_cover_data[index][0], self.non_cover_data[index][1])

    def specialize(self, complex, xn, xs):
        """
        Funkcja wykonująca specjalizację AQ.

        Args:
            complex: List[bitarray.bitarray] → lista selektorów
            xs: List[bitarray.bitarray] → lista zawierająca wartości atrybutów (1 z n) ziarna pozytywnego
            xn: List[bitarray.bitarray] → lista zawierająca wartości atrybutów (1 z n) ziarna negatywnego

        Return:
            List[List[bitarray.bitarray]] → wyspecjalizowane kompleksy
        """
        new_complexes = []
        for i, (att_xn, att_xs) in enumerate(zip(xn, xs)):
            if att_xn != att_xs:
                new_complex = deepcopy(complex)
                new_complex[i] = new_complex[i] ^ att_xn
                new_complexes.append(new_complex)
        return new_complexes

    def remove_non_maximum_general_complexes(self, complexes):
        """
        Funkcja usuwająca ze zbioru kompleksów wszystkie kompleksy, dla których w zbiorze kompleksow istnieje kompleks bardziej ogólny.

        Args:
            complexes: List[List[bitarray.bitarray]] → lista kompleksów
        Return:
            List[List[bitarray.bitarray]] → lista zawierająca kompleksy maksymalnie ogólne
        """
        to_remove = []
        for i, complex_i in enumerate(complexes):
            for j, complex_j in enumerate(complexes):
                if i != j and self.check_if_complex_is_more_general(complex_j, complex_i):
                    to_remove.append(complex_i)
                    break
        for complex_to_remove in to_remove:
            complexes.remove(complex_to_remove)
        return complexes

    def check_if_complex_is_more_general(self, complex_1, complex_2):
        """
        Funkcja sprawdzająca, czy pierwszy kompleks jest bardziej ogólny niż drugi kompleks.
        Do sprawdzenia wykorzystywana jest funkcja bitowa and.
        Args:
            complex_1: List[bitarray.bitarray] → lista selektorów pierwszego kompleksu
            complex_2: List[bitarray.bitarray] → lista selektorów drugiego kompleksu
        """
        return all((sel_1 & sel_2) == sel_2 for sel_1, sel_2 in zip(complex_1, complex_2))

    def select_best_m_complexes(self, m, complexes, correct_class):
        """
        Funkcja pozostawia w zbiorze kompleksów m najlepszych kompleksów.
        Jakość kompleksów oceniana jest jako liczba przykładów poprawnie
        sklasyfikowanych pomniejszoną o liczbę przykładów źle sklasyfikowanych.
        Do oceny brane są przykłady pochodzące ze zbioru non_cover_data.

        Args:
            m: int → parametr określający, ile najlepszych kompleksów pozostaje w zbiorze kompleksów; wystąpienie tego argumentu w tej funkcji, pomimo atrybutu self.m jest zmotywowane wybraniem jednego najlepszego kompleksu na końcu procesu tworzenia nowej reguły
            complexes: List[List[bitarray.bitarray]] → lista kompleksów
            correct_class: Any → klasa ziarna pozytywnego

        Return:
            List[List[bitarray.bitarray]] → lista m najlepszych kompleksów
        """
        complex_rates = []
        for complex in complexes:
            rate = 0
            for x, x_class in self.non_cover_data:
                if self.check_covering(complex, x):
                    if x_class == correct_class:
                        rate += 1
                    else:
                        rate -= 1
            complex_rates.append((rate, complex))
        complex_rates.sort(reverse=True, key=lambda x: x[0])
        best_m_complexes = [complex for _, complex in complex_rates[:m]]
        return best_m_complexes


    def check_covering(self, complex, data):
        """
        Funkcja sprawdza, czy kompleks pokrywa przykład.

        Args:
            complex: List[bitarray.bitarray] → lista selektorów kompleksu
            data: List[bitarray.bitarray] → lista zawierająca zakodowane wartości atrybutów przykładu
        Return:
            bool → True - jeśli kompleks pokrywa przykład, False w przeciwnym przypadku
        """
        return all((sel & att) == att for sel, att in zip(complex, data))

    def classify(self, x):
        """
        Funkcja klasyfikująca zadany przykład. W pierwszej kolejności wartości atrybutów przykłądu kodowane są do przestrzeni 1 z n
        W przypadku, gdy przykład pokrywany jest przez kilka reguł,
        jako predykowaną klasę uznawana jest ta, która wystąpiła najwięcej razy w tych regułach.
        Jeżeli żadna reguła nie pokrywa zadanego przykładu zwracana jest wartość None.

        Args:
            x: numpy.ndarray[Any] → lista zawierająca tylko wartości atrybutów przykładu

        Return:
            Any → predykcja klasy dla przykładu
        """
        x_one_hot = []
        for i, att in enumerate(x):
                x_one_hot.append(self.encode_att_to_one_hot(att, i))
        class_counter = {}
        for complex, cla in self.rules:
            if self.check_covering(complex, x_one_hot):
                class_counter[cla] = class_counter.setdefault(cla, 0) + 1
        return max(class_counter, key=class_counter.get, default=None)



