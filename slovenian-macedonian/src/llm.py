import torch
import numpy as np

from transformers import BertModel, BertTokenizer

class IExtractor:
    def __init__(self):
        return
    
    def avg_word_embeddings(self, embeddings: dict):
        return

    def get_embeddings(self, sentences, batch_size=16, layer_index=5, reduced_dim=128):
        return


class BERTExtractor(IExtractor):
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)

        # Move the model to GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def avg_pool_embeddings(self, embeddings, target_dim):
        """Apply average pooling to reduce the dimensionality of embeddings."""
        current_dim = embeddings.shape[-1]
        factor = current_dim // target_dim
        if current_dim % target_dim != 0:
            raise ValueError("Target dimension should be a divisor of the current dimension.")
        
        pooled_embeddings = torch.nn.functional.avg_pool1d(embeddings.unsqueeze(0), kernel_size=factor, stride=factor)
        return pooled_embeddings.squeeze(0)

    def avg_word_embeddings(self, embeddings: dict):
        if len(embeddings) == 0:
            return None
        return np.mean(list(embeddings.values()), axis=0)

    def get_embeddings(self, sentences, batch_size=16, layer_index=5, reduced_dim=128):
        embeddings_dict = {}
        # Process sentences in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
            
            # Move input tensors to the GPU
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            hidden_states = outputs.hidden_states[layer_index]
            
            for sentence, input_ids, hidden_state in zip(batch, inputs['input_ids'], hidden_states):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                for token, embedding in zip(tokens, hidden_state):
                    if token not in embeddings_dict:
                        embeddings_dict[token] = []
                    reduced_embedding = self.avg_pool_embeddings(embedding, reduced_dim)
                    embeddings_dict[token].append(reduced_embedding.cpu().numpy())
        
        # Average the embeddings for each token
        averaged_embeddings_dict = {token: torch.mean(torch.tensor(embeddings), dim=0).numpy() for token, embeddings in embeddings_dict.items()}
        return averaged_embeddings_dict
