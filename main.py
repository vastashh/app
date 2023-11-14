from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import LlamaCpp
import json
import whisper
from pydub import AudioSegment
from datetime import timedelta
import re

# Директория для сохранения мп3
mp3_dir = "/app/splitted_mp3"
# Проверяем, существует ли директория, если нет - создаем
os.makedirs(mp3_dir, exist_ok=True)
MODEL_PATH = "/home/inserv/application/models/model-q8_0.gguf"

app = FastAPI()


# Глобальные переменные
llm = None
model = None
global SYSTEM_PROMPT




def save_upload_file(file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        buffer.write(file.file.read())


def split_audio(file_path: str, path_to_save: str):
    # Загрузите двухканальный MP3-файл
    audio = AudioSegment.from_mp3(file_path)
    # Разделите аудио на два канала
    audio = audio.split_to_mono()
    operator_channel = audio[0]
    client_channel = audio[1]
    filename = os.path.basename(file_path)
    filename = os.path.splitext(filename)[0]
    # Сохраните каждый канал в отдельный файл
    os.makedirs(path_to_save + '/' + filename, exist_ok=True)
    new_file_path = os.path.join(path_to_save + '/' + filename, filename + "_op.mp3")
    operator_channel.export(new_file_path, format="mp3")
    new_file_path = os.path.join(path_to_save + '/' + filename, filename + "_cl.mp3")
    client_channel.export(new_file_path, format="mp3")


def get_dialog(splitted_dialog_path: str):
    cl_file = '_cl.mp3'
    op_file = '_op.mp3'
    for file in os.listdir(splitted_dialog_path):
        if file.endswith(cl_file):
            cl_file_path = os.path.join(splitted_dialog_path, file)
        if file.endswith(op_file):
            op_file_path = os.path.join(splitted_dialog_path, file)
    result_op = model.transcribe(op_file_path, language="ru")
    result_cl = model.transcribe(cl_file_path, language="ru")
    result_op = result_op['segments']
    result_cl = result_cl['segments']
    dialog = []
    for segment in result_op:
        startTime = str(0) + str(timedelta(seconds=int(segment['start'])))
        person = 'operator'
        text = segment['text']
        text = text[1:] if text[0] == ' ' else text
        segment = (startTime, person, text)
        dialog.append(segment)
    for segment in result_cl:
        startTime = str(0) + str(timedelta(seconds=int(segment['start'])))
        person = 'client'
        text = segment['text']
        text = text[1:] if text[0] == ' ' else text
        segment = (startTime, person, text)
        dialog.append(segment)

    sorted_dialog = sorted(dialog, key=lambda x: x[0])

    return sorted_dialog


def combine_messages(sorted_dialog: list):
    result = []
    current_tuple = None

    for item in sorted_dialog:
        if current_tuple is None:
            current_tuple = item
        elif item[1] in ('client', 'operator'):
            if current_tuple[1] == item[1]:
                current_tuple = (current_tuple[0], current_tuple[1], current_tuple[2] + ' ' + item[2])
            else:
                result.append(current_tuple)
                current_tuple = item

    if current_tuple is not None:
        result.append(current_tuple)

    sorted_dialog = result
    return sorted_dialog


def make_dialog(messages):
    dialog = ''
    for i in messages:
        if i[1] == 'client':
            dialog += ('Клиент: ' + i[2] + '\n')
        if i[1] == 'operator':
            dialog += ('Оператор: ' + i[2] + '\n')
    return (dialog)


ending_beginning_text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=0,
)


def extract_mark_from_string(string: str):
    """Функция достает оценку от 1 до 10 из строки"""

    substring_to_remove = "из 10"
    result_string = string.replace(substring_to_remove, "")
    # Используем регулярное выражение для поиска
    matches = re.findall(r'\b(?:[0-9]|10)\b', result_string)
    # Если есть оценки
    if matches:
        mark = matches[0]
        return mark
    else:
        return None


def process_dialog(content: str):
    if len(content) > 1000:
        # Начало диалога
        beginning = ending_beginning_text_splitter.split_text(content)[0]
        print(beginning)
        # Конец- переворачиваем чтоб 1000 с конца отрезать
        ending = ending_beginning_text_splitter.split_text(content[::-1])[0][::-1]
        print(ending)
    else:
        beginning = content
        ending = beginning

    SYSTEM_PROMPT = "Ответь подробно: что хотел спросить клиент у оператора в этом диалоге?"
    beginning = my_generate(beginning)
    SYSTEM_PROMPT = "Ответь подробно: чем помог оператор клиенту?"
    ending = my_generate(ending)
    SYSTEM_PROMPT = "Оцени релевантность ответа от 1 до 10 по описанию начала и конца диалога."
    question = 'Начало диалога: ' + beginning + '\n' + 'Конец диалога: ' + ending
    response = my_generate(question)
    score = extract_mark_from_string(response)
    return beginning, ending, score


# Код при запуске сервера
@app.on_event("startup")
async def startup_event():
    global llm, model
    n_gpu_layers = 200
    n_batch = 512

    # Загрузка llm
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repeat_penalty=1.1,
        verbose=True)
    # Загрузка whisper

    model = whisper.load_model(name='large', download_root='/home/inserv/application/models/whisper')




# Для модели---------------------------------------------
SYSTEM_PROMPT = ""
SYSTEM_TOKEN = 1587
USER_TOKEN = 2188
BOT_TOKEN = 12435
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)


def my_generate(
        text,
        llm=llm,
        n_ctx=2048,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repeat_penalty=1.1
):
    '''Переписанная функция для генерации ответа'''
    model = llm.client

    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    user_message = text
    message_tokens = get_message_tokens(model=model, role="user", content=user_message)
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens

    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty
    )
    answer = ''
    for token in generator:
        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break
        answer += token_str
    return answer
# -------------------------------------------------------

@app.post("/relevance/")
async def relevance(file: UploadFile = File(...)):
    file_path = os.path.join(mp3_dir, file.filename)
    save_upload_file(file, file_path)
    split_audio(file_path, mp3_dir)
    splitted_audio_path = mp3_dir + '/' + file.filename
    not_combined = get_dialog(splitted_audio_path)
    #удаляем файлы
    files = os.listdir(mp3_dir)
    # Проходим по каждому файлу и удаляем его
    for file in files:
        file_path = os.path.join(mp3_dir, file)
        os.remove(file_path)
    combined = combine_messages(not_combined)
    dialog = make_dialog(combined)
    beginning, ending, score = process_dialog(dialog)

    return JSONResponse(content={"beginning": beginning, "ending": ending, "score": score})