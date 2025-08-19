# pdf-to-db

## Setup

** Must have 2025AA-full UMLS 30gb local data.
** Must use mamba for package management.

0. Use semantic_types_definitions_and_cui.py <MRCONSO.RRF> <MRSTY.RRF> <MRDEF.RRF> <output_file> to create a "joined.txt" file containing combined filed.
1. Use mamba create --name <environment_name> python=3.12.10 and create 2 environments in different terminals one for requirements_ep.txt and one for requirements.
2. Use mamba install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz to install the medium model of scispacy.
3. Ensure that ollama is running llama3.1:8b. Use ollama pull llama3.1 --> ollama run llama3.1:8b
4. Have access to UMLS. Create a .env and paste in UMLS_API_KEY = no quotes around api key
5. Currently you need to run each python file one at a time.

## Commands 
1. convert PDF to Markdown and markdown into chunked json: python uni-lm.py "pdf of choice"
2. convert chunked json into entities: python entity_processor1.py --> entities json
