"""
"""


__all__ = [
    "ATTACK_RECIPES",
    "RECOMMENDED_RECIPES",
    "BUILTIN_DATASETS",
    "DATASET_NAME_MAPPING",
    "MODEL_DEFAULT_DATASET",
]


# attack recipes
ATTACK_RECIPES = {
    "a2t": "A2T",
    "bae": "BAE",
    "bert_attack": "BERTAttack",
    "checklist": "CheckList",
    "clare": "CLARE",
    "deep_word_bug": "DeepWordBug",
    "fd": "FD",
    "faster_genetic": "FasterGenetic",
    "genetic": "Genetic",
    "hotflip": "HotFlip",
    "iga": "IGA",
    "input_reduction": "InputReduction",
    "kuleshov": "Kuleshov",
    "pruthi": "Pruthi",
    "pso": "PSO",
    "pwws": "PWWS",
    "text_bugger": "TextBugger",
    "text_fooler": "TextFooler",
    "viper": "VIPER",
}

RECOMMENDED_RECIPES = [
    "a2t",
    "bae",
    "bert_attack",
    "deep_word_bug",
    "fd",
    "hotflip",
    "pwws",
    "text_bugger",
    "text_fooler",
    "viper",
]


# datasets
BUILTIN_DATASETS = {
    "amazon": "AmazonReviewsZH",
    "dianping": "DianPingTiny",
    "imdb": "IMDBReviewsTiny",
    "jd_binary": "JDBinaryTiny",
    "jd_full": "JDFullTiny",
    "sst": "SST",
    "ifeng": "Ifeng",
    "chinanews": "Chinanews",
}

DATASET_NAME_MAPPING = {
    "amazon": "amazon_reviews_zh",
    "dianping": "dianping_tiny",
    "imdb": "imdb_reviews_tiny",
    "jd_binary": "jd_binary_tiny",
    "jd_full": "jd_full_tiny",
    "sst": "sst",
    "ifeng": "ifeng",
    "chinanews": "chinanews",
}

MODEL_DEFAULT_DATASET = {
    "bert_amazon_zh": "amazon_reviews_zh",
    "roberta_chinanews": "chinanews",
    "roberta_dianping": "dianping_tiny",
    "roberta_ifeng": "ifeng",
    "roberta_sst": "sst",
}


# loggers related
ATTACK_ARGS_TO_LOG = [
    "ignore_errors",
    "language",
    "num_examples",
    "num_successful_examples",
    "max_len",
    "random_seed",
    "shuffle",
    "subset",
    "time_out",
    "query_budget",
    "robust_threshold",
    # the following are not in AttackArgs
    "model",
    "model_path",
    "dataset",
    "recipe",
]
