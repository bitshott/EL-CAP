from pathlib import Path
import torch

class MLMConfig():
    SCIBERT_MODEL: str = "allenai/scibert_scivocab_uncased"
    PUBMED_BERT_MODEL: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" 
    
    CSV_PATH: str = '/mnt/Supermicro/data2/chembl_35/chembl_passages.csv'
    PICKLE_PATH: str = '/mnt/Supermicro/data2/chembl_35/chembl_passages.pkl'
    TEXT_COL: str = 'passage'
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODELS_OUTPUT: Path = BASE_DIR / 'models'

    GRAD_ACCUM: int = 2
    EPOCHES: int = 5
    BATCH_PER_DEVICE: int = 16
    WARMUP_RATIO: float = 0.05
    LR: float = 5e-5

    RANDOM_SEED: int = 42
    TOKEN_MAX_LENGTH: int = 256

class CLConfig():
    TEXT_ENCODER: str = 'bitshott/scibert_scivocab_chembl_passages_v1'
    DEVICE: str = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    EPOCHES: int = 5
    LR: float = 2e-5
    L2_NORM: float = 1e-4
    GNN_HIDDEN_DIMS: list[int] = [9, 32, 64, 128, 256]

    RANDOM_SEED: int = 42



cl_config = CLConfig()
mlm_config = MLMConfig()

