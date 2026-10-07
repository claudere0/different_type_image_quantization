import numpy as np
from PIL import Image
import pytest
import io

def quantize_bpc(data: np.ndarray, bits: int) -> np.ndarray:
    """Функция квантизации битовой глубины."""
    levels = (2 ** bits) - 1
    normalized = data / 255.0
    quantized = np.round(normalized * levels)
    restored = (quantized / levels) * 255.0
    return np.clip(restored, 0, 255).astype(np.uint8)

def test_bpc_quantization_levels():
    # Создаём градиент от 0 до 255
    dummy_img = np.arange(256, dtype=np.uint8).reshape((16, 16))
    
    # 1 бит должен оставлять только 2 уникальных значения: 0 и 255
    res_1bit = quantize_bpc(dummy_img, bits=1)
    unique_1bit = np.unique(res_1bit)
    assert len(unique_1bit) <= 2
    assert set(unique_1bit).issubset({0, 255})
    
    # 2 бита: максимум 4 уникальных уровня
    res_2bit = quantize_bpc(dummy_img, bits=2)
    assert len(np.unique(res_2bit)) <= 4

def test_jpeg_compression_file_size():
    # Создаём цветное тестовое изображение 100x100
    img = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
    
    buf_high = io.BytesIO()
    img.save(buf_high, format="JPEG", quality=95)
    size_high = buf_high.tell()
    
    buf_low = io.BytesIO()
    img.save(buf_low, format="JPEG", quality=25)
    size_low = buf_low.tell()
    
    # Сжатие q25 обязано быть меньше q95
    assert size_low < size_high