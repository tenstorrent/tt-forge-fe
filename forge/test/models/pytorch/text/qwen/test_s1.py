import torch
import torch.nn as nn
import forge
from forge.verify.verify import verify
from test.utils import download_model
from transformers import AutoModel,AutoTokenizer
from loguru import logger
from test.models.models_utils import get_detailed_instruct

def test_decoder():
    class decoder(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.l = model.layers[0]
            
        def forward(self,hidden_states,position_ids,cache_position,position_embeddings_0,position_embeddings_1 ):
    
            logger.info("self.l={}",self.l)
            logger.info("hidden_states={}",hidden_states)
            logger.info("hidden_states.shape={}",hidden_states.shape)
            logger.info("hidden_states.dtype={}",hidden_states.dtype)
            logger.info("position_embeddings_0={}",position_embeddings_0)
            logger.info("position_embeddings_1={}",position_embeddings_1)
            logger.info("position_embeddings[0].shape={}",position_embeddings_0.shape)
            logger.info("position_embeddings[0].dtype={}",position_embeddings_0.dtype)
            logger.info("position_embeddings[1].shape={}",position_embeddings_1.shape)
            logger.info("position_embeddings[1].dtype={}",position_embeddings_1.dtype)
            logger.info("position_ids={}",position_ids)
            logger.info("cache_position={}",cache_position)

            inputs_embeds = self.l(hidden_states,None,position_ids,None,False,False,cache_position,(position_embeddings_0,position_embeddings_1))
            return inputs_embeds

    variant = "Qwen/Qwen3-Embedding-0.6B"

    framework_model = download_model(AutoModel.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()
    
    has_trainable_params = False
    for name, param in framework_model.named_parameters():
        if param.requires_grad:
            has_trainable_params = True
            logger.info(f"Trainable parameter found: {name}")
            param.requires_grad = False  # Freeze it

    if has_trainable_params:
        logger.info("Trainable parameters found and frozen.")
    else:
        logger.info("No trainable parameters found.")


    model = decoder(framework_model)
    model.eval()
    

    hidden_states = torch.load('hidden_states.pt')
    position_ids = torch.arange(31).unsqueeze(0)
    cache_position =  torch.arange(31).unsqueeze(0)
    position_embeddings_0 = torch.load('position_embeddings_0.pt')
    position_embeddings_1 = torch.load('position_embeddings_1.pt')
    
    inputs = [hidden_states, position_ids,cache_position,position_embeddings_0,position_embeddings_1]
    

    # Compile model
    compiled_model = forge.compile(model, inputs)

    # Model Verification
    verify(inputs, model, compiled_model)
