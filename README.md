# YaDisk-to-Text
API для получения краткой выжимки из содержания видео по ссылке на Яндекс.Диск

api - Главная директория, содержащая сам API.
.gitignore - Список файлов и разрешений, игнорируемых системой контроля версий
requirements.txt - Файл содержащей ссылки на внешние зависимости.

api/config - Параметры для запуска API.
api/models - Модели для работы с API.
api/main - Главный файл API.

api/utils - Директория, содержащая все классы и функции, используемые для VTT задачи
api/utils/__init__.py - Функции для работы с файловой системой.
api/utils/ml.py - Классы для работы с ИИ.

Для запуска API:
```bash
pip install -r requirements.txt
cd api
uvicorn main:app
```

ВНИМАНИЕ! На слабом железе обработка запросов работает долго.