# Code Summarization

## Web App

Welcome to the Code Summarizer, a web application powered by the [UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) machine learning model. This app enables you to easily generate concise summaries for your code snippets. Simply input your code, and the UniXcoder model will provide you with a clear and succinct summary of its functionality.

### Features:

- Code Summarization: Obtain quick and meaningful summaries for your code to enhance understanding and readability.

- Web Interface: User-friendly web-based interface for seamless interaction.

- Request History: All your code summarization requests are stored in the history section, allowing you to revisit and review previous summaries.

### How to Set Up

- `pip install poetry` -- install poetry
- `poetry install` -- install project with dependencies

Or using Docker:
- `docker build -t code-summarization-app .` -- build image
- `docker run -p 80:80 code-summarization-app` -- run container

### How to Use:
#### Fine-tune UniXcoder
- `poetry run wandb login <API-KEY>` -- set up api key for wandb logging
- `poetry run unixcoder-train`
#### Web-App
- `poetry run start` -- the URL to the web app will be provided in console
- Running container in Docker will create an app on localhost 80 port.


- Navigate to the web app.
- Input your code snippet in the provided text area.
- Click "Submit" to receive a summary.
- Explore the history section to review past code summarization requests.

## Code Summarization problem

### Example
```python
def write_json(data,file_path):
    data = json.dumps(data)
    with open(file_path, 'w') as f:
        f.write(data)
```

Proposed result:

```python
['Write JSON to file', 'Write json to file', 'Write a json file']
```

### Data Download for fine-tuning

Download [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) dataset

```bash
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
rm dataset.zip
cd dataset
wget https://zenodo.org/record/7857872/files/python.zip
wget https://zenodo.org/record/7857872/files/java.zip
wget https://zenodo.org/record/7857872/files/ruby.zip
wget https://zenodo.org/record/7857872/files/javascript.zip
wget https://zenodo.org/record/7857872/files/go.zip
wget https://zenodo.org/record/7857872/files/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```

### Fine-Tune Setting

For set up different fine-tune settings modify [finetune.yaml](src/unixcoder_pipeline/training/configs/finetune.yaml) or add new config to [src/unixcoder_pipeline/training/configs](src/unixcoder_pipeline/training/configs)
