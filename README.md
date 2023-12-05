# Code Summarization

## Web App

Welcome to the Code Summarizer, a web application powered by the UniXcoder machine learning model. This app enables you to easily generate concise summaries for your code snippets. Simply input your code, and the UniXcoder model will provide you with a clear and succinct summary of its functionality.

### Features:

- Code Summarization: Obtain quick and meaningful summaries for your code to enhance understanding and readability.

- Web Interface: User-friendly web-based interface for seamless interaction.

- Request History: All your code summarization requests are stored in the history section, allowing you to revisit and review previous summaries.

### How to Use:

- `poetry run start` -- the URL to the web app will be provided in console.
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

### Data Download

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

Here we provide fine-tune settings for code summarization, whose results are reported in the paper.

```shell
lang=python

# Training
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/$lang/train.jsonl \
	--dev_filename dataset/$lang/valid.jsonl \
	--output_dir saved_models/$lang \
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 10 \
	--train_batch_size 48 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 
	
# Evaluating	
python src/training/run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename dataset/$lang/test.jsonl \
	--output_dir saved_models/$lang \
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 10 \
	--train_batch_size 48 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 	
```
