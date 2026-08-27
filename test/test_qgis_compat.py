from qgis_compat import unpack_vector_writer_result


def test_unpack_vector_writer_two_value_result():
    assert unpack_vector_writer_result((0, '')) == (0, '')


def test_unpack_vector_writer_four_value_result():
    assert unpack_vector_writer_result(
        (0, '', '/tmp/output.gpkg', 'layer')) == (0, '')


def test_unpack_vector_writer_scalar_result():
    assert unpack_vector_writer_result(0) == (0, '')
