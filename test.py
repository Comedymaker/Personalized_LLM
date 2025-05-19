#Large Languase Model, such as llama2-7b.
from calflops import calculate_flops
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models.tokenizer import Tokenizer
from models.model import TinyModelLoader, LargeModelLoader

batch_size = 4
max_seq_length = 2
tokenizer = Tokenizer.load_tokenizer()
model = LargeModelLoader.load_model()
model.resize_token_embeddings(len(tokenizer))
flops, macs, params = calculate_flops(model=model,
                                      input_shape=(batch_size, max_seq_length),
                                      transformer_tokenizer=tokenizer)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# FLOPs:54.83 TFLOPS   MACs:27.41 TMACs   Params:14.16 B 