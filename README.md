# Fraud Detection API

API на FastAPI для оцінювання ймовірності шахрайства у фінансових транзакціях. Підтримує одиничні та пакетні запити, перевіряє вхідні дані й записує транзакції з високим ризиком у журнал.

Модель — pipeline scikit-learn із масштабуванням числових ознак, One-Hot кодуванням категорій та логістичною регресією. Вона навчається на 2500 синтетичних транзакціях із фіксованим seed `42`. Це демонстраційний проєкт; якість прогнозів на реальних транзакціях не оцінювалася.

## Встановлення та запуск

Потрібні Python і pip. Локальне середовище проєкту використовує Python 3.14.

Виконайте команди з кореня репозиторію:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
mkdir -p logs
python -m uvicorn src.main:app --reload
```

У Windows для активації середовища використовуйте `venv\Scripts\activate`; каталог `logs` створіть командою `mkdir logs`, якщо він відсутній.

Навчання створює `models/model.joblib`. API завантажує цей файл під час запуску, тому модель потрібно підготувати до запуску сервера. Шляхи до моделі та журналу відносні: запускайте сервер із кореня репозиторію. Після повторного навчання перезапустіть сервер, щоб завантажити нову модель.

Після запуску доступні:

- API: <http://127.0.0.1:8000>
- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>

## Ендпоінти

| Метод | Шлях | Призначення |
| --- | --- | --- |
| GET | `/health` | Перевірка стану API та завантаження моделі |
| POST | `/v1/predict` | Оцінювання однієї транзакції |
| POST | `/v1/predict/batch` | Оцінювання масиву транзакцій, максимум 50 за запит |

### Вхідні дані

| Поле | Тип | Обмеження |
| --- | --- | --- |
| `user_id` | integer | Більше 0 |
| `amount` | number | Більше 0, не більше 100 000 |
| `merchant_category` | string | `grocery`, `electronics`, `clothing`, `entertainment`, `travel`, `other` |
| `location_lat` | number | Від −90 до 90 |
| `location_lon` | number | Від −180 до 180 |
| `is_international` | boolean | Необов'язкове; за замовчуванням `false` |

API приймає категорію `clothing`, хоча синтетична навчальна вибірка її не містить. Кодер обробляє невідомі категорії через `handle_unknown="ignore"`.

### Перевірка стану

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

### Одиничний прогноз

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 42,
    "amount": 1500.50,
    "merchant_category": "grocery",
    "location_lat": 50.45,
    "location_lon": 30.52,
    "is_international": false
  }'
```

Приклад структури відповіді (значення ілюстративні):

```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-10-06T12:00:00Z",
  "is_fraud": false,
  "fraud_probability": 0.12,
  "risk_level": "LOW"
}
```

`transaction_id` — згенерований UUID, `timestamp` — час відповіді в UTC, `fraud_probability` — оцінена ймовірність шахрайства. `is_fraud` дорівнює `true`, якщо ймовірність ≥ 0.5.

| Рівень ризику | Ймовірність |
| --- | --- |
| `LOW` | Менше 0.3 |
| `MEDIUM` | Від 0.3 до менш ніж 0.7 |
| `HIGH` | Від 0.7 включно |

Транзакції з рівнем `HIGH` записуються у фоновому завданні до `logs/suspicious_transactions.log` у форматі `transaction_id,fraud_probability`, по одному запису на рядок. Каталог `logs` має існувати й бути доступним для запису.

### Пакетний прогноз

```bash
curl -X POST http://127.0.0.1:8000/v1/predict/batch \
  -H 'Content-Type: application/json' \
  -d '[
    {"user_id": 42, "amount": 1500.50, "merchant_category": "grocery", "location_lat": 50.45, "location_lon": 30.52},
    {"user_id": 43, "amount": 95000, "merchant_category": "travel", "location_lat": 48.85, "location_lon": 2.35, "is_international": true}
  ]'
```

Відповідь — масив прогнозів у порядку вхідних транзакцій. Надсилайте від 1 до 50 транзакцій: обробка порожнього масиву в поточній реалізації не передбачена.

### Помилки

- `400` — пакет містить понад 50 транзакцій.
- `422` — вхідні дані не відповідають схемі.
- `503` — модель відсутня у стані запущеного застосунку.

Якщо файл `models/model.joblib` відсутній або не завантажується, запуск застосунку завершується помилкою ще до обробки запитів.

## Тести

```bash
python -m pytest
```

Тести перевіряють стан API, одиничний і пакетний прогнози, валідацію полів та обмеження розміру пакета. Якщо модель відсутня, тести автоматично запускають її навчання.

## Структура проєкту

```text
.
├── src/
│   ├── main.py          # FastAPI, завантаження моделі, прогнози та журналювання
│   └── schemas.py       # Схеми запитів і відповідей, категорії продавців
├── tests/
│   └── test_main.py     # Тести API
├── models/
│   └── model.joblib     # Згенерована модель; виключена з Git
├── logs/               # Журнал транзакцій із високим ризиком
├── train_model.py      # Генерація синтетичних даних і навчання
└── requirements.txt    # Залежності
```
