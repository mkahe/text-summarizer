FROM python:3.8.10 


# Install dependencies
RUN pip install --upgrade pip

# Copy the current directory contents into the container at /app
COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en-core-web-trf


# open the port 5002
EXPOSE 5002

CMD [ "python app.py" ]