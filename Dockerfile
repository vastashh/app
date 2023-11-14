# Используем базовый образ с поддержкой Python
FROM python:3.9

# Устанавливаем зависимости
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
# Копируем приложение в образ
COPY . /app

# Устанавливаем рабочую директорию
WORKDIR /app

# Добавляем инструкцию VOLUME для предоставления доступа к папке
VOLUME ["/home/inserv/application/models"]

# Открываем порт, который будет слушать FastAPI
EXPOSE 8080

# Запускаем FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]




