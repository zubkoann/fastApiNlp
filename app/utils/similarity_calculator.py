import textdistance


def calculate_similarity(method: str, line1: str, line2: str) -> float:
    try:
        similarity_func = getattr(textdistance, method)
    except AttributeError:
        # Если метод не найден, поднимаем ошибку
        raise ValueError(f"Unsupported method: {method}")

    # Вычисляем схожесть используя выбранный метод
    similarity = similarity_func(line1, line2)
    return similarity
