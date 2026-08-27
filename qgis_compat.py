"""Small compatibility helpers for QGIS API return-value differences."""


def unpack_vector_writer_result(writer_result):
    """Return ``(error, message)`` from QGIS two- or four-value results."""
    if isinstance(writer_result, (tuple, list)):
        error = writer_result[0] if writer_result else None
        message = writer_result[1] if len(writer_result) > 1 else ''
        return error, message
    return writer_result, ''
