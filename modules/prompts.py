import pandas as pd

def enhance_objects_in_designs(data: pd.DataFrame, batch_size: int):
    prompts = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        prompt = """
        You are an expert extraction algorithm for numismatic design descriptions.
        Your goal is to enhance the list of identified objects in the following designs.
        You will be provided with a design description and a list of objects, and you will output JSON objects containing the following information:

        {
            design_id: int, // Unique identifier of the design
            new_list_of_strings: [(string, string)] // Enhanced list of objects in the form of tuples: (entity, class)
        }

        Focus on identifying all semantically meaningful objects within each design,
        according to the categories "PERSON", "OBJECT", "ANIMAL", "PLANT".
        Do not include terms that describe the coin itself or are redundant.
        Only consider significant elements of the design.
        Consider each design as distinct.
        Remove any objects that are less significant or redundant in the context of the design description,
        but mention each entity at least once if it contributes to the overall meaning.

        For example, in a design of "Nude Aphrodite standing facing, head right, holding her breast with right hand and pudenda with left hand; to left, Eros, seated on a dolphin downwards.":

        - "hand" is less significant because it is a part of the actions (holding) rather than a standalone meaningful object in the design, but if "hand" is holding something important, include it.
        - "head" is also less significant as it is a common part of the figure and does not add unique semantic value in this context unless it is specifically referenced in the design.

        Example:
        {
            design_id: 36, // Unique identifier of the design
            Original Design: "Nude Aphrodite standing facing, head right, holding her breast with right hand and pudenda with left hand; to left, Eros, seated on a dolphin downwards.",
            Original List of Strings: [("Aphrodite", "PERSON"), ("head", "OBJECT"), ("breast", "OBJECT"), ("hand", "OBJECT"), ("hand", "OBJECT"), ("Eros", "PERSON"), ("dolphin", "ANIMAL")],
            Enhanced List of Strings: [("Aphrodite", "PERSON"), ("breast", "OBJECT"), ("pudenda", "OBJECT"), ("Eros", "PERSON"), ("dolphin", "ANIMAL")]
        }

        Example:
        {
            design_id: 8, // Unique identifier of the design
            Original Design: "Prize amphora on ornamental stand; within linear square and incuse square.",
            Original List of Strings: [("amphora", "OBJECT")],
            Enhanced List of Strings: [("amphora", "OBJECT"), ("stand", "OBJECT")]
        }

        Example:
        {
            design_id: 101, // Unique identifier of the design
            Original Design: "Asclepius resting on left, on wing serpent to right.",
            Original List of Strings: [("Asclepius", "PERSON"), ("serpent", "ANIMAL")],
            Enhanced List of Strings: [("Asclepius", "PERSON"), ("serpent", "ANIMAL"), ("wing", "OBJECT")]
        }

        For each design, provide the enhanced list of strings in the form of tuples: [(entity, class), ...].

        Respond only with the following fields for each design:
        {
            design_id: int, // Unique identifier of the design
            new_list_of_strings: [(string, string)] // Enhanced list of objects in the form of tuples: (entity, class)
        }

        Now, enhance the following designs:
        """
        for _, entry in batch.iterrows():
            prompt += f"""
            {{
                design_id: {entry['id']}, // Unique identifier of the design
                Original Design: "{entry['design_en']}",
                Original List of Strings: {entry['list_of_strings']}
            }},
            """
        prompt += """
        Notes: 
        - Objects should be atomic and not compound terms. For example, horn of ammon should be represented as the key "horn" with the class "OBJECT".
        - Persons should be named as they are, like Alexander the Great or Antoninus Pius, and not as "Alexander", "Great" or "Antoninus", "Pius".
        
        Respond with the design_id and the enhanced list of strings in valid JSON format for each design, like this:
        {
            "design_id": 36,
            "new_list_of_strings": [("Aphrodite", "PERSON"), ("breast", "OBJECT"), ("pudenda", "OBJECT"), ("Eros", "PERSON"), ("dolphin", "ANIMAL")]
        }
        """
        prompts.append(prompt)

    return prompts



def validate_overall_objects_in_designs(data: pd.DataFrame, batch_size: int):
    prompts = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        prompt = """
        You are an expert validator algorithm for numismatic design descriptions.
        Your task is to classify the overall likelihood of the identified objects list in the following designs.
        You will be provided with a design description and a list of objects, and you will output JSON objects containing the following information:

        {
            design_id: int, // Unique identifier of the design
            relevance: int, // Rating: 1 (very relevant), 0 (somewhat relevant), -1 (irrelevant)
            correctness: int, // Rating: 1 (very correct), 0 (somewhat correct), -1 (incorrect)
            comment_enh: string // Concise comment on the enhanced list
        }

        Rate the entire new list of objects for each design based on the following criteria:
        - Relevance (1: very relevant, 0: somewhat relevant, -1: irrelevant): Are the objects relevant and meaningful in the context of the Design?
        - Correctness (1: very correct, 0: somewhat correct, -1: incorrect): Are the identified objects correctly classified and named based on the Design?

        Consider each design as distinct.

        Examples:
        {
            design_id: 36, // Unique identifier of the design
            Original List of Strings: [["Aphrodite", "PERSON"], ["head", "OBJECT"], ["breast", "OBJECT"], ["hand", "OBJECT"], ["Eros", "PERSON"], ["dolphin", "ANIMAL"]],
            Enhanced List of Strings: [["Aphrodite", "PERSON"], ["breast", "OBJECT"], ["pudenda", "OBJECT"], ["Eros", "PERSON"], ["dolphin", "ANIMAL"]],
            Design: "Nude Aphrodite standing facing, head right, holding her breast with right hand and pudenda with left hand; to left, Eros, seated on a dolphin downwards.",
            relevance: 1,
            correctness: 1,
            comment_enh: "The enhanced list includes all significant objects mentioned in the design description and has improved relevance by including 'pudenda' and excluding redundant 'head' and 'hand'."
        },
        {
            design_id: 8,
            Original List of Strings: [["amphora", "OBJECT"]],
            Enhanced List of Strings: [["amphora", "OBJECT"], ["stand", "OBJECT"]],
            Design: "Prize amphora on ornamental stand; within linear square and incuse square.",
            relevance: 1,
            comment_enh: "The enhanced list includes the main objects of the design."
        },
        {
            design_id: 101,
            Original List of Strings: [["athlete", "PERSON"]],
            Enhanced List of Strings: [["athlete", "PERSON"], ["head", "OBJECT"]],
            Design: "Athlete holding a discus with both hands.",
            relevance: 0,
            correctness: -1,
            comment_enh: "The list misses the 'discus' which is an important object in the design description. The 'head' is not part of the context of the design."
        },
        {
            design_id: 202,
            Original List of Strings: [["figure", "PERSON"], ["border", "OBJECT"]],
            Enhanced List of Strings: [["figure", "PERSON"], ["border", "OBJECT"]],
            Design: "Figure of a man surrounded by a decorative border.",
            relevance: -1,
            correctness: 0,
            comment_enh: "The list is missing several key objects such as 'man' and includes less relevant ones like 'border'."
        }

        For each design, provide the ratings for relevance, and correctness along with any comments.

        Respond only with the following fields for each design:
        {
            design_id: int, // Unique identifier of the design
            relevance: int, // Rating: 1 (very relevant), 0 (somewhat relevant), -1 (irrelevant)
            correctness: int, // Rating: 1 (very correct), 0 (somewhat correct), -1 (incorrect)
            comment_enh: string // Comment on the enhanced list
        }

        Now, validate the following designs:
        """
        for _, entry in batch.iterrows():
            prompt += f"""
            {{
                design_id: {entry['design_id']}, // Unique identifier of the design
                Original List of Strings: {entry['list_of_strings']},
                Enhanced List of Strings: {entry['new_list_of_strings']},
                Design: "{entry['design_en']}"
            }},
            """
        prompt += """
        Notes: 
        - Objects should be atomic and not compound terms. For example, horn of ammon should be represented as the key "horn" with the class "OBJECT".
        - Persons should be named as they are, like Alexander the Great or Antoninus Pius, and not as "Alexander", "Great" or "Antoninus", "Pius".
        

        Respond with the design_id, relevance, correctness, and comment_enh in valid JSON format for each design, like this:
        {
            "design_id": 36,
            "relevance": 1,
            "correctness": 1,
            "comment_enh": "The enhanced list includes all significant objects mentioned in the design description and has improved relevance by including 'pudenda' and excluding redundant 'head' and 'hand'."
        }
        """
        prompts.append(prompt)

    return prompts



def find_subject_object_pairs_prompts(data: pd.DataFrame, batch_size: int):
    prompts = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        prompt = """
        You are an expert extraction algorithm for numismatic design descriptions.
        Extract all semantically meaningful pairs of entities from the following designs.
        Each "List of Strings" entry contains tuples in the form of [(entity, class), ...]
        for the classes "PERSON", "OBJECT", "ANIMAL", "PLANT".
        For each possible pair, provide a new row with the Design ID.

        Ensure to consider all entities for pairing, but only include pairs that are semantically meaningful based on the design description.

        Example:
        {
            design_id: 478, // Unique identifier of the design
            Design: "Eros seated_on right on dolphin, holding rein in both hands; dolphin holding oar. Border of dots.",
            List of Strings: [("Eros", "PERSON"), ("rein", "OBJECT"), ("dolphin", "ANIMAL"), ("oar", "OBJECT")]
        }

        Expected Response:
        [
            {"design_id": 478, "s_o_id": "a", "s": "Eros", "subject_class": "PERSON", "o": "dolphin", "object_class": "ANIMAL"},
            {"design_id": 478, "s_o_id": "b", "s": "Eros", "subject_class": "PERSON", "o": "rein", "object_class": "OBJECT"},
            {"design_id": 478, "s_o_id": "c", "s": "dolphin", "subject_class": "ANIMAL", "o": "oar", "object_class": "OBJECT"}
        ]

        Example:
        {
            design_id: 67, // Unique identifier of the design
            Design: "Apollo and Artemis in front of large conical torch; to left, Apollo standing right, seen from behind, holding spear in left arm and chlamys over left shoulder grasping right hand with Artemis, to right, standing left, wearing a short chiton and boot, holding a bow in left hand.",
            List of Strings: [("Apollo", "PERSON"), ("Artemis", "PERSON"), ("torch", "OBJECT"), ("spear", "OBJECT"), ("chlamys", "OBJECT"), ("chiton", "OBJECT"), ("boot", "OBJECT"), ("bow", "OBJECT")]
        }

        Expected Response:
        [
            {"design_id": 67, "s_o_id": "a", "s": "Apollo", "subject_class": "PERSON", "o": "torch", "object_class": "OBJECT"},
            {"design_id": 67, "s_o_id": "b", "s": "Apollo", "subject_class": "PERSON", "o": "spear", "object_class": "OBJECT"},
            {"design_id": 67, "s_o_id": "c", "s": "Apollo", "subject_class": "PERSON", "o": "chlamys", "object_class": "OBJECT"},
            {"design_id": 67, "s_o_id": "d", "s": "Apollo", "subject_class": "PERSON", "o": "Artemis", "object_class": "PERSON"},
            {"design_id": 67, "s_o_id": "e", "s": "Artemis", "subject_class": "PERSON", "o": "torch", "object_class": "OBJECT"},
            {"design_id": 67, "s_o_id": "f", "s": "Artemis", "subject_class": "PERSON", "o": "bow", "object_class": "OBJECT"},
            {"design_id": 67, "s_o_id": "g", "s": "Artemis", "subject_class": "PERSON", "o": "chiton", "object_class": "OBJECT"},
            {"design_id": 67, "s_o_id": "h", "s": "Artemis", "subject_class": "PERSON", "o": "boot", "object_class": "OBJECT"}
        ]

        Now, process the following designs:
        """
        for _, entry in batch.iterrows():
            prompt += f"""
            {{
                design_id: {entry['design_id']}, // Unique identifier of the design
                Design: "{entry['design_en']}",
                List of Strings: {entry['new_list_of_strings']}
            }},
            """
        prompt += """
        Respond only with the following fields for each possible pair of entities:
        {
            "design_id": int, // Unique identifier of the design
            "s_o_id": string, // Identifier for each pair
            "s": string, // Subject entity
            "subject_class": string, // Class of the subject entity
            "o": string, // Object entity
            "object_class": string // Class of the object entity
        }

        Note: Ensure the following when identifying entity pairs:
        - Each pair must be semantically meaningful based on the design description.
        - Avoid redundant pairs.
        - Double-check the semantic meaning and correctness of each pair.
        - Check for meaningful Subject-Object, Object-Object, and Subject-Subject pairs.
        - Ensure all entities are considered for pairing.
        - If no meaningful pairs are found for a Design ID, use "NULL" for the subject, subject_class, object, and object_class fields.
        - If the same pair is mentioned multiple times, provide a new row with a different s_o_id \in {a, b, c, d, ...}.

        Respond only with the dictionary entries excluding 'design_en' and in valid JSON format, like this:
        [
            {
                "design_id": 114, 
                "s_o_id": "a", 
                "s": "Artemis", 
                "subject_class": "PERSON", 
                "o": "torch", 
                "object_class": "OBJECT"
            },
            {
                "design_id": 90, 
                "s_o_id": "a", 
                "s": "NULL", 
                "subject_class": "NULL", 
                "o": "NULL", 
                "object_class": "NULL"
            }
        ]
        """
        prompts.append(prompt)

    return prompts



def validate_subject_object_pairs(data: pd.DataFrame, batch_size: int):
    prompts = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        prompt = """
        You are an expert validator algorithm for numismatic design descriptions. 
        Your task is to classify the validity of the identified subject-object, or entity pairs, in the following designs.
        Rate each pair of objects based on the following criteria:
        - Validity (1: valid, 0: questionable, -1: invalid): Is the pair semantically meaningful and correct in the context of the design description?

        Example:
        {
            design_id: 478, // Unique identifier of the design
            Design: "Eros seated_on right on dolphin, holding rein in both hands; dolphin holding oar. Border of dots.",
            Subject-Object Pairs: [
                {"design_id": 478, "s_o_id": "a", "s": "Eros", "subject_class": "PERSON", "o": "dolphin", "object_class": "ANIMAL"},
                {"design_id": 478, "s_o_id": "b", "s": "Eros", "subject_class": "PERSON", "o": "rein", "object_class": "OBJECT"},
                {"design_id": 478, "s_o_id": "c", "s": "Eros", "subject_class": "PERSON", "o": "hands", "object_class": "OBJECT"},
                {"design_id": 478, "s_o_id": "d", "s": "dolphin", "subject_class": "ANIMAL", "o": "oar", "object_class": "OBJECT"},
                {"design_id": 478, "s_o_id": "e", "s": "rein", "subject_class": "OBJECT", "o": "hands", "object_class": "OBJECT"}
            ]
        }

        Ratings:
        [
            {"design_id": 478, "s_o_id": "a", "validity_sop": 1, "comment_sop": "Correct and meaningful pair."},
            {"design_id": 478, "s_o_id": "b", "validity_sop": 1, "comment_sop": "Correct and meaningful pair."},
            {"design_id": 478, "s_o_id": "c", "validity_sop": -1, "comment_sop": "Not relevant pair as 'hands' of a person is not significant."},
            {"design_id": 478, "s_o_id": "d", "validity_sop": 1, "comment_sop": "Correct and meaningful pair."},
            {"design_id": 478, "s_o_id": "e", "validity_sop": 0, "comment_sop": "Questionable pair as 'rein' and 'hands' are not significant."}
        ]

        For each design, provide the validity rating for each subject-object pair along with a short comment.

        Respond only with the following fields for each design:
        {
            design_id: int, // Unique identifier of the design
            s_o_id: string, // Identifier for each pair
            validity_sop: int, // Validity rating: 1 (valid), 0 (questionable), -1 (invalid)
            comment_sop: string // Short comment on the validity of the pair
        }

        Now, validate the following designs:
        """
        for _, entry in batch.iterrows():
            prompt += f"""
            {{
                design_id: {entry['design_id']}, // Unique identifier of the design
                s_o_id: "{entry['s_o_id']}",
                Design: "{entry['design_en']}"
                Subject: "{entry['s']}" ({entry['subject_class']}),
                Object: "{entry['o']}" ({entry['object_class']}),
            }},
            """
        prompt += """
        Notes: 
        - Do not validate the single entities (entity classes), only focus on the correctness of the pairs of entities (subject, object pairs) based on the Design.

        Respond with the design_id, s_o_id, validity_sop, and comment_sop in valid JSON format, like this:
        {
            "design_id": 478, 
            "s_o_id": "a", 
            "validity_sop": 1, 
            "comment_sop": "Correct and meaningful pair."
        }
        """
        prompts.append(prompt)

    return prompts


def find_predicates_prompts(data: pd.DataFrame, batch_size: int):
    prompts = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        prompt = """
        You are an expert relation extraction algorithm for numismatic design descriptions.
        Extract the most likely predicate (action or state) for each subject-object pair relation from the following designs.
        The predicate should be explicitly mentioned in the design description and should be the most likely relation between the subject and object.
        Ensure that the predicate is an action or state, not an entity or object.

        Examples:
        {
            design_id: 478, // Unique identifier of the design
            SOP Id: "a",
            Subject: "Eros", "subject_class": "PERSON",
            Object: "dolphin", "object_class": "ANIMAL",
            Design: "Eros seated_on right on dolphin, holding rein in both hands; dolphin holding oar. Border of dots."
        }

        Expected Response:
        {"design_id": 478, "s_o_id": "a", "predicate": "seated_on"},
        
        {
            design_id: 478, // Unique identifier of the design
            SOP Id: "b",
            Subject: "dolphin", "subject_class": "ANIMAL",
            Object: "oar", "object_class": "OBJECT",
            Design: "Eros seated_on right on dolphin, holding rein in both hands; dolphin holding oar. Border of dots."
        }

        Expected Response:
        {"design_id": 478, "s_o_id": "b", "predicate": "holding"},

        {
            design_id: 50, // Unique identifier of the design
            SOP Id: "a",
            Subject: "Apollo", "subject_class": "PERSON",
            Object: "Wreath", "object_class": "OBJECT",
            Design: "Wreath head of Apollo, right."
        }

        Expected Response:
        {"design_id": 50, "s_o_id": "a", "predicate": "NULL"},

        Now, process the following designs:
        """
        for _, entry in batch.iterrows():
            prompt += f"""
            {{
                design_id: {entry['design_id']}, // Unique identifier of the design
                SOP Id: {entry['s_o_id']},
                Subject: {entry['s']} ({entry['subject_class']}),
                Object: {entry['o']} ({entry['object_class']}),
                Design: "{entry['design_en']}"
            }},
            """
        prompt += """
        Respond only with the following fields for each subject-object pair:
        {
            design_id: int, // Unique identifier of the design
            s_o_id: string, // Identifier for each pair
            predicate: string // The most likely action or state relation
        }

        Note: Ensure the following when identifying predicates:
        - Each predicate must be an action or state explicitly mentioned in the design description.
        - Avoid using entities or objects as predicates.
        - Extract the most likely relation predicate for each subject-object pair.
        - Double-check the semantic meaning and correctness of each predicate.
        - If no predicate is found, use "NULL" for the predicate field.

        Respond only with the dictionary entries design_id, s_o_id, and predicate in valid JSON format, like this:
        [
            {
                "design_id": 114, 
                "s_o_id": "a", 
                "predicate": "containing"
            },
            {
                "design_id": 50, 
                "s_o_id": "a", 
                "predicate": "NULL"
            }
        ]
        """
        prompts.append(prompt)

    return prompts


def validate_spo_triples(data: pd.DataFrame, batch_size: int):
    prompts = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        prompt = """
        You are an expert validator algorithm for numismatic design descriptions.
        Your task is to evaluate the validity of the extracted subject-predicate-object (SPO) triples in the following designs.
        Rate each SPO triple based on the following criteria:
        - Validity (1: valid, 0: questionable, -1: invalid): Is the triple semantically meaningful and correct in the context of the design description?
        - Explicitness: Is the predicate explicitly mentioned in the design description?

        If the identified predicate is not plausible or is NULL, provide an implicit predicate based on the design description 
        and alaso validate this predicate with -1.

        Examples:
        {
            design_id: 478, // Unique identifier of the design
            SOP Id: "a",
            Subject: "Eros", subject_class: "PERSON",
            Predicate: "seated_on",
            Object: "dolphin", object_class: "ANIMAL",
            Design: "Eros seated_on right on dolphin, holding rein in both hands; dolphin holding oar. Border of dots."
        }

        Ratings:
        [
            {
                design_id: 478,
                s_o_id: "a",
                validity_pred: 1,
                comment_pred: "Correct and meaningful SPO triple.",
                implicit_pred: "NULL"
            }
        ]

        {
            design_id: 478, // Unique identifier of the design
            SOP Id: "b",
            Subject: "dolphin", subject_class: "ANIMAL",
            Predicate: "holding",
            Object: "oar", object_class: "OBJECT",
            Design: "Eros seated_on right on dolphin, holding rein in both hands; dolphin holding oar. Border of dots."
        }

        Ratings:
        [
            {
                design_id: 478,
                s_o_id: "b",
                validity_pred: 1,
                comment_pred: "Correct and meaningful SPO triple.",
                implicit_pred: "NULL"
            }
        ]

                {
            design_id: 53, // Unique identifier of the design
            SOP Id: "a",
            Subject: "Apollo", subject_class: "PERSON",
            Predicate: "NULL",
            Object: "Wreath", object_class: "OBJECT",
            Design: "Wreath head of Apollo, right; scallop below."
        }

        Ratings:
        [
            {
                design_id: 53,
                s_o_id: "a",
                validity_pred: -1,
                comment_pred: "NULL is not a valid predicate.",
                implicit_pred: "wearing"
            }
        ]

        Now, validate the following designs:
        """
        for _, entry in batch.iterrows():
            prompt += f"""
            {{
                design_id: {entry['design_id']}, // Unique identifier of the design
                SOP Id: {entry['s_o_id']},
                Subject: {entry['s']} ({entry['subject_class']}),
                Predicate: {entry['predicate']},
                Object: {entry['o']} ({entry['object_class']}),
                Design: "{entry['design_en']}"
            }},
            """
        prompt += """
        Respond only with the following fields for each SPO triple:
        {
            design_id: int, // Unique identifier of the design
            s_o_id: string, // Identifier for each SPO triple
            validity_pred: int, // Validity rating: 1 (valid), 0 (questionable), -1 (invalid)
            comment_pred: string, // Short comment on the validity of the triple
            implicit_pred: string // Implicit predicate if the identified predicate is not plausible or is NULL
        }

        Respond with the design_id, s_o_id, validity_pred, comment_pred, and implicit_pred in valid JSON format, like this:
        {
            "design_id": 478, 
            "s_o_id": "a", 
            "validity_pred": 1, 
            "comment_pred": "Correct and meaningful SPO triple.", "implicit_pred": "NULL"
        }
        """
        prompts.append(prompt)

    return prompts
