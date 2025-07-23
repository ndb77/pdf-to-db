# pdf-to-db

## Setup
1. Use python -m spacy download en_core_sci_sm to download the entity identifier
2. Use python -m venv /path/to/new/virtual/environment to open 3 new venvs in 3 different terminals. Use pip install -r requirements_"python-file-name".txt
3. Ensure that ollama is running llama3.2. Use ollama pull llama3.2 --> ollama run llama3.2
4. Have access to UMLS. Create a .env and paste in UMLS_API_KEY = no quotes around api key
5. Currently you need to run each python file one at a time - intent is to make it easier to debug with simple integration ability into single file at the end.
6. Currently temperature is set to .1, consider temperature of 0 to make more deterministic.

Next steps, add vectorization.


## Commands 
1. convert PDF to Markdown: python marker_pdf_converter "pdf of choice"
2. convert markdown into chunked json: python chunke "name of markdown"
3. convert chunked json into labelled entities: python entity_extraction "name of gnerated json"
