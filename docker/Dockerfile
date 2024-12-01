FROM python:3.9

WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN pip install numpy transformers onnxruntime flask

#COPY ./data/model_repository/ ./data/model_repository/
COPY ./dir_configs.py ./
COPY ./src/models/transformers_model/inference_exported_model.py ./inference_exported_model.py
COPY ./src/models/transformers_model/app.py ./app.py
COPY ./docker/entrypoints/entrypoint_flask_app.sh ./

RUN chmod +x ./entrypoint_*.sh

CMD /app/entrypoint_flask_app.sh

