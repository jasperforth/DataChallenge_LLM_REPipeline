import sys
import pandas as pd
import ast
import logging
import tqdm
from pathlib import Path
from dataclasses import dataclass
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure that the correct path to the `cnt` module is appended to sys.path
# Adjusting the path to ensure `cnt` can be found
repo_dir = Path().resolve()  # Use current directory for relative pathing

# Append the appropriate directory for modules
sys.path.append(str(repo_dir / 'libs' / 'NLP_on_multilingual_coin_datasets'))

# Import necessary modules from `cnt`
from NLP_on_multilingual_coin_datasets.cnt.io import Database_Connection
from NLP_on_multilingual_coin_datasets.cnt.annotate import annotate_designs
# from NLP_on_multilingual_coin_datasets.cnt.model import load_ner_model_v2
from NLP_on_multilingual_coin_datasets.cnt.preprocess import Preprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class PreprocessingConfig:
    id_col: str = "id"
    design_col: str = "design_en"
    use_lemma_stem: bool = False
    language: str = "_en"
    add_columns: list = None
    csv_path: Path = Path("./data/source/lists/csv")
    csv_designs_filename: str = "annotated_designs.csv"
    json_path: Path = Path("./data/results/json")
    tmp_path: Path = Path("./data/results/tmp")
    database: str = "nlp_challenge"

    def __post_init__(self):
        if self.add_columns is None:
            self.add_columns = ["id", "name" + self.language, "alternativenames" + self.language]

class LoadingPreprocessedDesigns():
    def __init__(self, dc: Database_Connection, PreprocessingConfig: PreprocessingConfig):
        self.dc = dc
        self.prep_cfg = PreprocessingConfig

        self.id_col = self.prep_cfg.id_col
        self.design_col = self.prep_cfg.design_col

    def load_designs_csv_or_process_database(self):
        csv_filepath = f"{self.prep_cfg.csv_path}/{self.prep_cfg.csv_designs_filename}"
        logging.info(f"Checking if file {csv_filepath} exists.")
        if Path(csv_filepath).is_file():
            df_designs = pd.read_csv(csv_filepath)
            if type(df_designs.annotations.iloc[0]) == str:
                logging.info("Converting annotations to list.")
                df_designs['annotations'] = df_designs['annotations'].apply(ast.literal_eval)
            logging.info("File exists and was loaded.")
        else:
            logging.info("File does not exist. Loading from database and running preprocessing.")
            df_designs = self.preprocess_designs()

            Path(self.prep_cfg.csv_path).mkdir(parents=True, exist_ok=True)
            df_designs.to_csv(csv_filepath, index=False)
            logging.info(f"Preprocessed designs saved to {csv_filepath}.")

        return df_designs

    def preprocess_designs(self):
        logging.info("Starting preprocessing of designs.")
        try:
            df_designs_raw = self.dc.load_designs_from_db("nlp_training_designs", 
                                                          [self.id_col, 
                                                           self.design_col])

            entities = self.load_entities()

            annotated_designs = annotate_designs(entities, df_designs_raw, self.id_col, self.design_col)
            annotated_designs = annotated_designs[annotated_designs.annotations.map(len) > 0]

            annotated_designs["design_en_changed"] = ""
            df_entities = self.dc.load_from_db("nlp_list_entities", self.prep_cfg.add_columns)

            preprocess = self.initialize_preprocess(df_entities)
            annotated_designs = self.clean_design_names(annotated_designs)

            logging.info("Applying defined preprocessing rules to design names.")
            annotated_designs["design_en_changed"] = annotated_designs.swifter.apply(
                lambda row: preprocess.preprocess_design(row.design_en, row.id)[0], axis=1)
            logging.info("Completed applying preprocessing rules to design names.")
            
            logging.info("Deleting brackets and question marks from design names.")
            annotated_designs["design_en_changed"] = annotated_designs.swifter.apply(
                lambda row: row["design_en_changed"].replace("?", "").replace("(", "").replace(")", ""), axis=1)
            logging.info("Completed deleting brackets and question marks from design names.")

            # Renaming columns
            annotated_designs.rename(
                columns={"design_en": "design_en_orig", "design_en_changed": "design_en", "annotations": "annotations_orig"},
                inplace=True)
            
            # Re-annotate the cleaned designs
            _designs = annotate_designs(entities, annotated_designs[["id", "design_en"]], self.id_col, self.design_col)
            _designs = _designs[_designs.annotations.map(len) > 0]

            # Merging the annotations back
            # _designs['annotations'] = _designs['annotations'].apply(ast.literal_eval)
            annotated_designs = annotated_designs.merge(_designs[["id", "annotations"]], on="id")

            logging.info("Preprocessing completed successfully.")
            return annotated_designs
        
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def load_entities(self):
        database = self.prep_cfg.database
        add_columns = self.prep_cfg.add_columns
        entities = {}
        try:
            for entity_type in ["PERSON", "OBJECT", "ANIMAL", "PLANT"]:
                entities[entity_type] = self.dc.load_entities_from_db_v2(
                    f"nlp_list_ent", entity_type, add_columns, [add_columns[1]], ",", True)
                logging.info(f"Loaded {entity_type} entities successfully.")
            return entities
        except Exception as e:
            logging.error(f"Error loading entities: {e}")
            raise

        

    def initialize_preprocess(self, df_entities: pd.DataFrame):
        preprocess = Preprocess()
        preprocess.add_rule("horseman", "horse man")
        preprocess.add_rule("horsemen", "horse men")

        logging.info("Adding rules from entities.")
        for _, row in tqdm.tqdm(df_entities.iterrows(), total=df_entities.shape[0], desc="Initializing Preprocess"):
            if row["alternativenames_en"] is not None:
                standard_name = row["name_en"]
                alt_names = row["alternativenames_en"].split(", ")
                for alt_name in alt_names:
                    preprocess.add_rule(alt_name, standard_name)

        for rule in list(preprocess.rules):
            if " I." in rule or " II." in rule or " III." in rule or " IV." in rule or " V." in rule:
                del preprocess.rules[rule]
        
        logging.info("Completed adding rules from entities.")
        return preprocess


    def clean_design_names(self, annotated_designs):
            logging.info("Cleaning design names.")
            for index, row in annotated_designs.iterrows():
                if " I." in row["design_en"]:
                    annotated_designs.at[index, "design_en"] = row["design_en"].replace(" I.", " I")
                if " II." in row["design_en"]:
                    annotated_designs.at[index, "design_en"] = row["design_en"].replace(" II.", " II")
                if " III." in row["design_en"]:
                    annotated_designs.at[index, "design_en"] = row["design_en"].replace(" III.", " III")
                if " IV." in row["design_en"]:
                    annotated_designs.at[index, "design_en"] = row["design_en"].replace(" IV.", " IV")
                if " V." in row["design_en"]:
                    annotated_designs.at[index, "design_en"] = row["design_en"].replace(" V.", " V")

            logging.info("Completed cleaning design names.")
            return annotated_designs