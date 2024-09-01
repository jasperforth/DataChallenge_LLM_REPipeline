# DataChallenge_LLM_REPipeline

**Description**:  
DataChallenge_LLM_REPipeline is a pipeline leveraging LLMs to enhance relation extraction, based on a multilingual coin dataset, and integrating the Frankfurt-BigDataLab [NLP-on-multilingual-coin-datasets](https://github.com/Frankfurt-BigDataLab/NLP-on-multilingual-coin-datasets) submodule.

## Installation

To clone this repository with the necessary submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/yourrepo.git
```

If you have already cloned the repository without initializing submodules, you can set them up by running:

```bash
git submodule update --init --recursive
```

### Setting up the project

After cloning the repository, run the following command to set up the necessary symbolic links:

```bash
python setup_symlink.py
```

## Requirements

- python          3.8.12
- SQL Database 

The project provides both a `requirements.txt` for `pip` and an `environment.yml` for `conda`.

### Conda

To create the environment with Conda:

```bash
conda env create -f environment.yml
conda activate data_challenge_git
```

### Pip
To install dependencies with pip:
```bash
pip install -r requirements.txt
```
Ensure your environment matches the specified versions to avoid compatibility issues.


## Usage

### API Configuration

- The API version and usage details will be provided later.
- Ensure you use your own API key, which should be stored securely in an `.env` file (this file is already included in `.gitignore` to protect your credentials).

### Example Usage

Examples and usage instructions will be added after relevant files are transferred and cleaned. The planned examples include:

1. **Evaluating and Exploring Generated Results (No SQL Database Required):**
   - A Jupyter notebook designed to evaluate and compare the generated Relation Extraction (RE) triples with the existing ground truth. This notebook serves as an evaluation tool to assess the quality of the generated data and includes comparisons with the ground truth data.

2. **Chat API Calls (SQL Database Required):**
   - Example usage of the chat LLM API, demonstrating how to interact with the model using data stored in the SQL database.

3. **BatchAPI Calls for OpenAI (gpt-4.0) (SQL Database Required):**
   - Instructions for using the batch API bot, specifically for OpenAI's GPT-4.0, including data retrieval from the SQL database.

#### Database Setup for Examples Requiring SQL 

- **Data Import:** Create the SQL database from the SQL dump located at `/data/source/data/nlp_challenge.sql`.
- **Environment Variables:** Update the `.env` file with your database connection details:
  - `DB_USER`: Username for the database.
  - `DB_PASSWORD`: Password for the database.
  - `DB_HOST`: Host where the database is running (e.g., `127.0.0.1`).
  - `DB_PORT`: Port number the database is listening on (e.g., `3306`).


## Documentation

The project documentation and presentation slides are available in the repository as a PDF file: `DataChallenge_NLP_final.pdf`.

## References

This project incorporates data and methodologies from the following sources:

- [Corpus Nummorum](https://www.corpus-nummorum.eu/)
- [Online Coins of the Roman Empire (OCRE)](http://numismatics.org/ocre/)
- Patricia Klinger, Sebastian Gampe, Karsten Tolle, Ulrike Peter (2018). "Semantic Search based on Natural Language Processing – a Numismatic example." Journal of Ancient History and Archaeology (JAHA) Vol 5 No. 3 (2018) 68-79. DOI: [10.14795/j.v5i3.334](http://jaha.org.ro/index.php/JAHA/article/view/334/244)

## License

**English:**

This project is licensed under the Creative Commons Attribution - NonCommercial - ShareAlike 3.0 Germany License.  
To view a copy of this license, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc-sa/3.0/de/).

**Deutsch:**

Dieses Projekt ist lizenziert unter der Creative Commons Attribution - NonCommercial - ShareAlike 3.0 Germany License.  
Um eine Kopie dieser Lizenz zu sehen, besuchen Sie [Creative Commons Lizenz](https://creativecommons.org/licenses/by-nc-sa/3.0/de/).

The full license text will be included in a separate `LICENSE` file in the repository.

