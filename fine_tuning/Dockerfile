FROM unsloth/unsloth:latest

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace

ENV PYTHONUNBUFFERED=1
ENV TRAINING_DATA_PATH=/workspace/data/training-questions.json
ENV OUTPUT_DIR=/workspace/outputs/question-rewriter-qwen3-0.6b

CMD ["python", "fine_tuning/train.py"]
