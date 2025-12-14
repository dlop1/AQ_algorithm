from AQ_alg import AQ
from bitarray import bitarray

domains = [['val_1', 'val_2', 'val_3'], ['val_1', 'val_2']]
aq = AQ(1, False, domains)
data_x = [['val_1', 'val_2'], ['val_2', 'val_1'], ['val_3', 'val_2']]
data_y = [['pos'], ['neg'], ['pos']]

def test_check_covering_positive():
    complex1 = bitarray('01101')
    complex2 = bitarray('01001')
    assert aq.check_covering(complex1, complex2)

def test_check_covering_negative():
    complex1 = bitarray('01001')
    complex2 = bitarray('11101')
    assert not aq.check_covering(complex1, complex2)

def test_one_hot_encoding():
    data = data_x[1]
    expected_ohe = [bitarray('010'), bitarray('10')]
    assert aq.encode_att_to_one_hot(data[0], 0) == expected_ohe[0]
    assert aq.encode_att_to_one_hot(data[1], 1) == expected_ohe[1]

def test_specialize():
    xs = [bitarray('0100'), bitarray('0001'), bitarray('0010')]
    xn = [bitarray('0001'), bitarray('0100'), bitarray('0010')]
    complex = [bitarray('0111'), bitarray('0101'), bitarray('1111')]
    expected_result = [[bitarray('0110'), bitarray('0101'), bitarray('1111')], [bitarray('0111'), bitarray('0001'), bitarray('1111')]]
    assert aq.specialize(complex, xn, xs) == expected_result

def test_check_if_complex_is_more_general_positive():
    complex1 = [bitarray('0111'), bitarray('0111'), bitarray('1111')]
    complex2 = [bitarray('0101'), bitarray('0101'), bitarray('1111')]
    assert aq.check_if_complex_is_more_general(complex1, complex2)

def test_check_if_complex_is_more_general_negative():
    complex1 = [bitarray('0111'), bitarray('0111'), bitarray('1111')]
    complex2 = [bitarray('1101'), bitarray('0111'), bitarray('1111')]
    assert not aq.check_if_complex_is_more_general(complex1, complex2)

def test_remove_non_maximum_general_complexes():
    complex1 = [bitarray('0111'), bitarray('0111'), bitarray('1111')]
    complex2 = [bitarray('1101'), bitarray('1001'), bitarray('1111')]
    complex3 = [bitarray('0101'), bitarray('0101'), bitarray('1010')]
    complex4 = [bitarray('0100'), bitarray('1001'), bitarray('1011')]
    complexes = [complex1, complex2, complex3, complex4]
    assert aq.remove_non_maximum_general_complexes(complexes) == [complex1, complex2]

def test_select_best_m_complexes():
    aq.get_data(data_x, data_y)
    correct_class = 'pos'
    complex1 = [bitarray('101'), bitarray('01')]
    complex2 = [bitarray('010'), bitarray('11')]
    complex3 = [bitarray('111'), bitarray('11')]
    complexes = [complex1, complex2, complex3]
    assert aq.select_best_m_complexes(1, complexes, correct_class) == [complex1]
    assert aq.select_best_m_complexes(2, complexes, correct_class) == [complex1, complex3]
    assert aq.select_best_m_complexes(3, complexes, correct_class) == [complex1, complex3, complex2]

