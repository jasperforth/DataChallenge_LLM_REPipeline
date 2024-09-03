# DataChallenge_LLM_REPipeline

**Description**:  
The `DataChallenge_LLM_REPipeline` is designed to enhance relation extraction using Large Language Models (LLMs) on a multilingual coin dataset. This pipeline integrates the Frankfurt-BigDataLab [NLP-on-multilingual-coin-datasets](https://github.com/Frankfurt-BigDataLab/NLP-on-multilingual-coin-datasets) submodule.

## Pipeline Overview

The pipeline consists of three major steps, each involving processing followed by validation and classification of the results. These steps are executed by an LLM instance using specific prompts and loaded data. The prompts and data loading logic are located in the `prompts.py` module. Each step has its own set of prompts with examples and processing logic, and they build upon each other to form the complete pipeline. The same steps apply to both the chat API and batch API; the difference lies in how the APIs handle the process—batch API prepares the entire job at once, while chat API iterates through the prepared prompts.

### Pipeline Steps

#### Step 0: Check for More Possible Subjects or Objects
- **Input:** Design description and a list of strings (entities).
- **Output:** Identified subjects and objects categorized as PERSON, OBJECT, ANIMAL, PLANT.

#### Step 0.1: Validate and Classify Enhanced Entities
- **Purpose:** Validation and classification of the entities identified in Step 0.

#### Step 1: Identify Subject-Object Pairs
- **Purpose:** Identify potential subject-object pairs from the input data.

#### Step 1.1: Validate and Classify Object-Subject Pairs
- **Purpose:** Validation and classification of the subject-object pairs identified in Step 1.

#### Step 2: Combine Subject-Predicate-Object
- **Input:** Design description, subject-object pairs, and possible predicates.
- **Output:** A list of subject-predicate-object triples.

#### Step 2.1: Validate and Classify Extracted Relations
- **Purpose:** Validation and classification of the relations extracted in Step 2.

## Installation

To clone the repository along with the necessary submodules, run:

```bash
git clone --recurse-submodules https://github.com/yourrepo.git
```

If you have already cloned the repository without initializing the submodules, initialize them using:

```bash
git submodule update --init --recursive
```

### Project Setup

After cloning, set up the required symbolic links by executing:

```bash
python setup_symlink.py
```

## Requirements

- Python 3.8.12
- SQL Database

Dependencies can be installed using either `pip` or `conda`, as detailed below.

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

Ensure that your environment matches the specified versions to prevent compatibility issues.

## Usage

### API Configuration

- API version and usage details will be provided later.
- Store your API key securely in an `.env` file, which is included in `.gitignore` to protect your credentials.

### Example Usage

The following examples and instructions are now available and ready to use:

1. **Evaluating and Exploring Generated Results (No SQL Database Required):**
   - A Jupyter notebook is available for evaluating and comparing generated Relation Extraction (RE) triples with existing ground truth data. This notebook serves as a tool to assess the quality of the generated data and includes necessary comparisons with the ground truth.

2. **Chat API Calls (SQL Database Required):**
   - Example usage of the chat LLM API, demonstrating interaction with the model using data stored in the SQL database.

3. **Batch API Calls for OpenAI (GPT-4.0) (SQL Database Required):**
   - Instructions for using the batch API bot specifically with OpenAI's GPT-4.0, including data retrieval from the SQL database.

#### Database Setup for SQL-Dependent Examples

- **Data Import:** Initialize the SQL database using the SQL dump located at `/data/source/data/nlp_challenge.sql`.
- **Environment Variables:** Update the `.env` file with your database connection details:
  - `DB_USER`: Username for the database.
  - `DB_PASSWORD`: Password for the database.
  - `DB_HOST`: Database host (e.g., `127.0.0.1`).
  - `DB_PORT`: Database port (e.g., `3306`).

## Documentation

Comprehensive project documentation and presentation slides are available in the repository as a PDF file: `DataChallenge_NLP_final.pdf`.

## Planned Improvements

- **Notebook Enhancements:** The provided notebooks will be further cleaned and commented for better readability.
- **Unit Tests:** Unit tests will be added following the cleanup.

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

The full license text is included in the `LICENSE` file in the repository.
