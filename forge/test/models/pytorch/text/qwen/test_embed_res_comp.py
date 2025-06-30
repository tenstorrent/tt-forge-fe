import torch
import torch.nn as nn
import forge
from forge.verify.verify import verify
from test.utils import download_model
from transformers import AutoModel,AutoTokenizer
from loguru import logger
from test.models.models_utils import get_detailed_instruct
from scipy.stats import pearsonr
 
def test_res_comp():
    class model_embed(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.embed_tokens = model.embed_tokens
            logger.info("self.embed_tokens={}",self.embed_tokens)
            
            
        def forward(self,input_ids ):
            logger.info("input_ids={}",input_ids)
            logger.info("input_ids.dtype={}",input_ids.dtype)
            logger.info("input_ids.shape={}",input_ids.shape)
            
            inputs_embeds = self.embed_tokens(input_ids)
            return inputs_embeds
 

    variant = "Qwen/Qwen3-Embedding-0.6B"
    
    framework_model = download_model(AutoModel.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    
    # prepare input
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    input_texts = queries + documents

    # Tokenize the input texts
    input_tokens = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"]]

    
    model = model_embed(framework_model)
    model.eval()
 
    # Compile model
    compiled_model = forge.compile(model, inputs)
    
    # output comparison
    with torch.no_grad():
        fw_out = model(*inputs)
        co_out = compiled_model(*inputs)

    fw_tensor = fw_out[0] if isinstance(fw_out, list) else fw_out
    co_tensor = co_out[0] if isinstance(co_out, list) else co_out
    input_ids_tensor = inputs[0]

    pad_id = tokenizer.pad_token_id
    mask = (input_ids_tensor != pad_id)

    for b in range(fw_tensor.shape[0]):
        for s in range(fw_tensor.shape[1]):
            if mask[b, s]:
                fw_vec = fw_tensor[b, s]
                co_vec = co_tensor[b, s]
                diff = (fw_vec - co_vec).abs().max()
                if diff > 1e-5:
                    print(f"⚠️ Diff at [B={b}, S={s}] = {diff.item()}")

    print("PCC (manual):", pearsonr(fw_tensor.flatten().detach().cpu().numpy(), co_tensor.flatten().detach().cpu().numpy())[0])


