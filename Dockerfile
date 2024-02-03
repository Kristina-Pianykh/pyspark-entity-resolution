FROM bitnami/spark:3.5 as runtime

WORKDIR /app/
COPY ./requirements.txt ./
USER root
RUN pip install -r requirements.txt
# CMD ["python", "-m", "document_bot.main"]
