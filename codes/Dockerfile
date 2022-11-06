FROM python:3.8.10 


# Install dependencies
RUN pip install --upgrade pip

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm
# RUN python -m spacy download en-core-web-trf

COPY nltk_download.py /app/nltk_download.py
RUN python nltk_download.py

COPY templates /app/templates
COPY __init__.py /app/__init__.py
COPY app.py /app/app.py
COPY text_miner.py /app/text_miner.py

RUN python text_miner.py --init


# open the port 5001
EXPOSE 5001

CMD [ "python" , "app.py" ]