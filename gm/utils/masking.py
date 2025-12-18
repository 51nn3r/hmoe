import torch


def create_padding_mask(sequences, pad_token_id=0):
    """
    Создает маску паддинга из индексов

    Args:
        sequences: [batch_size, seq_len] или [batch_size, seq_len, depth]
        pad_token_id: ID токена паддинга

    Returns:
        Маска формы [batch_size, 1, 1, seq_len] (0 = разрешено, -inf = запрещено)
    """

    mask = (sequences == pad_token_id).unsqueeze(1).unsqueeze(1)
    return mask.float().masked_fill(mask == 1, float('-inf'))


def create_causal_mask(seq_len, device=None):
    """
    Создает каузальную маску

    Args:
        seq_len: длина последовательности
        device: устройство

    Returns:
        Маска формы [1, 1, seq_len, seq_len] (0 = разрешено, -inf = запрещено)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf')).unsqueeze(0).unsqueeze(0)


def create_full_mask(seq_len, device=None):
    """
    Создает маску полного внимания (все видят всех)

    Args:
        seq_len: длина последовательности
        device: устройство

    Returns:
        Маска формы [1, 1, seq_len, seq_len] (все нули - все разрешено)
    """
    return torch.zeros(1, 1, seq_len, seq_len, device=device)


def create_sliding_window_mask(seq_len, window_size, device=None):
    """
    Создает маску скользящего окна

    Args:
        seq_len: длина последовательности
        window_size: размер окна
        device: устройство

    Returns:
        Маска формы [1, 1, seq_len, seq_len]
    """
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)

    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i + 1] = 0

    return mask.unsqueeze(0).unsqueeze(0)


def extend_mask_for_learnable_vectors(original_mask, num_learnable_vectors):
    """
    Расширяет маску для обучаемых векторов (без транзитивности)

    Args:
        original_mask: исходная маска [batch_size, seq_len, seq_len]
        num_learnable_vectors: количество обучаемых векторов

    Returns:
        Расширенная маска [batch_size, new_seq_len, new_seq_len]
    """
    dims = original_mask.shape
    new_seq_len = dims[-1] + num_learnable_vectors

    extended_mask = torch.full(dims[:-2] + (new_seq_len, new_seq_len),
                               float('-inf'), device=original_mask.device)

    # Обучаемые векторы видят всех
    extended_mask[..., :num_learnable_vectors, :] = 0

    # Обычные токены видят только друг друга (исходная маска)
    extended_mask[..., num_learnable_vectors:, num_learnable_vectors:] = original_mask

    return extended_mask


def extend_mask_full_attention(original_mask, num_vectors):
    """
    Расширяет маску с полным вниманием для дополнительных векторов

    Args:
        original_mask: исходная маска [batch_size, 1, seq_len, seq_len]
        num_vectors: количество добавляемых векторов

    Returns:
        Расширенная маска [batch_size, new_seq_len, new_seq_len]
    """
    batch_size, _, orig_seq_len, _ = original_mask.shape
    new_seq_len = orig_seq_len + num_vectors

    extended_mask = torch.zeros(batch_size, new_seq_len, new_seq_len,
                                device=original_mask.device)

    # Копируем исходную маску
    extended_mask[:, :, num_vectors:, num_vectors:] = original_mask

    # Все видят всех для новых векторов
    extended_mask[:, :, :num_vectors, :] = 0
    extended_mask[:, :, :, :num_vectors] = 0

    return extended_mask


def combine_masks(*masks):
    """
    Комбинирует несколько масок (логическое И)

    Args:
        masks: список масок одинаковой формы

    Returns:
        Комбинированная маска
    """
    if not masks:
        raise ValueError("Список масок не может быть пустым")

    combined = masks[0].clone()
    for mask in masks[1:]:
        combined = torch.min(combined, mask)

    return combined


# Примеры использования
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Тестовые данные
    seq_len = 4
    batch_size = 2
    input_ids = torch.tensor([
        [1, 2, 0, 0],  # С паддингом
        [1, 2, 3, 4]  # Без паддинга
    ], device=device)

    print("1. Маска паддинга:")
    padding_mask = create_padding_mask(input_ids, pad_token_id=0)
    print(padding_mask[:, 0])

    print("\n2. Каузальная маска:")
    causal_mask = create_causal_mask(seq_len, device)
    print(causal_mask[0])

    print("\n3. Полная маска:")
    full_mask = create_full_mask(seq_len, device)
    print(full_mask[0])

    print("\n4. Расширение для обучаемых векторов:")
    extended_mask = extend_mask_for_learnable_vectors(causal_mask, num_learnable_vectors=2)
    print(extended_mask[0])

    print("\n5. Комбинированная маска:")
    combined = combine_masks(causal_mask, padding_mask)
    print(combined, combined.shape)

    extended_combined = extend_mask_for_learnable_vectors(combined, num_learnable_vectors=2)
    print('\n6. extended_combined')
    print(extended_combined, extended_combined.shape)
