# Import necessary type hints for clearer code annotation
from typing import List, Tuple

# Import regular expressions for tokenizing the text
import re

# Import numpy for numerical and data manipulation tasks
import numpy as np

# Import seaborn for plotting training loss
import seaborn as sns

# Import plotly for interactive scatter plot visualization of word embeddings
import plotly.express as px
from plotly.offline import plot

# Import PyTorch for building and training the neural network
import torch
import torch.nn as nn
import torch.optim as optim

# Import PCA from sklearn for dimensionality reduction of the learned embeddings
from sklearn.decomposition import PCA

def set_seed(seed: int = 1):
    """
    Set the random seed for NumPy and PyTorch (both CPU and GPU)
    to ensure that experiments are reproducible.
    
    Parameters:
    seed (int): The seed value to be used for all random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the function to set the seed for reproducibility.
set_seed(42)

# Define a corpus of sentences covering various topics (nature, travel, pets, etc.)
corpus = [
    "I like reading books in the early morning, while the sun rises and the gentle rain taps on my window.",
    "Cats and dogs often play together, creating moments of joy that mix with the soft melody of music and laughter.",
    "I like to drink coffee while listening to music, letting my dreams wander on roads paved with adventure and quiet hope.",
    "The moon and stars shine over quiet houses, inspiring nature lovers who appreciate the simple beauty of the night sky.",
    "I like travel and adventure, where reading books under old trees and amidst blooming flowers fuels my wild dreams.",
    "Rain, sun, and music blend on quiet days, as I like to watch cats and dogs frolic in the lively garden.",
    "I like nature's simple pleasures, such as sipping coffee, reading books, and enjoying the dance of sun, rain, and dreams.",
    "The road to adventure is paved with books, music, and soft whispers of wind under shining stars and bright moons.",
    "I like the harmony of nature and city life, where cars pass quiet houses while the sun and moon share the sky.",
    "In my dreams, I like to travel with cats and dogs, exploring vast fields where rain and sun create magical scenes.",
    "I like music that tells of travel and adventure, where every note mirrors the rhythm of busy roads and quiet forests.",
    "Under soft rain, I like to read books while listening to music, letting nature's melody guide my gentle journey.",
    "I like the warmth of the sun, the cool touch of rain, and the steady beat of music that inspires my travels.",
    "Cats, dogs, and the hum of coffee machines create a peaceful vibe, as I like to read books in calm corners.",
    "I like watching the moon rise as stars twinkle above, while soft music and gentle rain fill the quiet night.",
    "I like adventure and travel, where books guide my journeys and nature, with trees and flowers, inspires my dreams.",
    "Music and coffee often join my reading sessions, as I like to explore stories about travel, love, and wild adventures.",
    "I like quiet moments under the open sky, where the sun, moon, and stars share their light with dreaming hearts.",
    "Cats and dogs run freely on winding roads, and I like to follow their trail, inspired by books and soft music.",
    "I like sunny days mixed with unexpected rain, when music fills the air and my heart dances with each new journey.",
    "In quiet moments, I like to sip coffee, read books, and listen to music, letting dreams of adventure take flight.",
    "I like to imagine journeys where cats, dogs, and friendly souls travel together on roads paved with books and tunes.",
    "The blend of sun, rain, and soft music creates a perfect backdrop for me as I like to explore nature and read books.",
    "I like to travel during warm days, where the light of the sun and the calm of rain mingle with vibrant music.",
    "I like nature's beauty, as cats and dogs wander under tall trees, and the sun and moon dance in perfect harmony.",
    "In the quiet evening, I like to read books with a cup of coffee, while gentle music plays softly in the background.",
    "I like to watch roads unfold before me, where adventure awaits and cats, dogs, and music create lively scenes.",
    "I like to mix the taste of coffee with the sound of rain, reading books that tell stories of travel and dreams.",
    "The mix of sun, rain, and soft music reminds me that I like the simple joys of nature, travel, and adventure.",
    "I like to see beauty in every moment, where cats and dogs play under the sun, and music stirs my hopeful heart.",
    "Under bright sun and gentle rain, I like to listen to music and read books, dreaming of travel and grand adventures.",
    "I like to spend quiet afternoons in nature, where winding roads lead me to moments with cats, dogs, and soft tunes.",
    "I like to let my mind wander through dreams, guided by books and the gentle blend of sun, rain, and melody.",
    "In cool morning light, I like to drink coffee while reading books, with cats and dogs playing outside in nature.",
    "I like the mix of travel and adventure, where the sun, moon, and stars light roads filled with inspiring books.",
    "I like to share moments of joy with friends, discussing music, books, and the playful antics of cats, dogs, and nature.",
    "In my quiet corner, I like to listen to music, sip coffee, and read books that tell of travel, dreams, and adventure.",
    "I like to explore the world with an open heart, where nature, cats, dogs, and music create a canvas of endless stories.",
    "I like the pleasure of reading books on rainy days, as the sun peeks through clouds and music fills the gentle air.",
    "I like to take long walks on winding roads, where every step is a journey with nature, music, and inspiring tales.",
    "I like to dream of adventures where travel meets quiet moments, and cats, dogs, and music paint scenes of pure joy.",
    "I like to mix warm coffee with gentle rain sounds, reading books that ignite dreams of travel and boundless wonder.",
    "I like the way the sun lights up the sky, as music, books, and nature come together for moments of peaceful bliss.",
    "I like to watch the moon and stars guide my journey, while cats, dogs, and soft music keep my spirit light.",
    "I like to explore new roads, where travel, magical books, and the harmony of music fill my day with adventure.",
    "I like to enjoy moments of silence with coffee, as gentle rain, soft music, and a good book complete my day.",
    "I like to follow winding roads that lead to adventure, where nature, music, and playful pets create joyful scenes.",
    "I like the balance of sun and rain, the quiet of nature, and the melody of music that accompanies my cherished journeys.",
    "I like to embrace each day with hope, where books, coffee, and soft music blend with adventures of travel and dreams.",
    "I like to create simple stories that mix nature, travel, cats, dogs, music, and the timeless magic of dreams."
]


def tokenize(document:str) -> List[str]:
    """
    Convert a document to lowercase and split it into tokens (words)
    using a regex that matches word characters.
    """
    return re.findall("\w+", document.lower())


def extract_vocabulary(corpus:List[str]) -> List[str]:
    """
    Create a sorted list of unique tokens (vocabulary) from the corpus.
    It tokenizes each document and then extracts the unique words.
    """
    vocabulary = [token for document in corpus for token in tokenize(document)]
    return sorted(list(set(vocabulary)))


def one_hot_encode(token:str, vocabulary:List[str]) -> List[float]:
    """
    One-hot encode a given token based on its position in the vocabulary list.
    Returns a list of floats where the index corresponding to the token is 1.0
    and all others are 0.0.
    """
    encoded_token = []
    for item in vocabulary:
        if token == item:
            encoded_token.append(1.0)
        else:
            encoded_token.append(0.0)
    return encoded_token


def get_skipped_context_pairs(corpus:List[str], size:int) -> Tuple[List[str]]:
    """
    Generate pairs of words for the skip-gram model.
    
    For each token in each document, the function considers a context window
    of given 'size' around the token (both before and after).
    It returns two lists:
      - skipped_tokens: the skipped words,
      - context_tokens: the words in the context of each skipped word.
    """
    skipped_tokens = []
    context_tokens = []
    for document in corpus:
        tokens = tokenize(document)
        for i, skipped in enumerate(tokens):
            
            # Look 'size' tokens to the left and right for context pairs
            for k in range(size):
                # Check left context, ensuring index is valid
                if i-k-1 > -1:
                    skipped_tokens.append(skipped)
                    context_tokens.append(tokens[i-k-1])
                
                # Check right context, ensuring index is valid
                if i+k+1 < len(tokens):
                    skipped_tokens.append(skipped)
                    context_tokens.append(tokens[i+k+1])
            
    return skipped_tokens, context_tokens
        

# Extract the vocabulary from the corpus
vocabulary = extract_vocabulary(corpus)

# Generate skip-gram pairs with a context window size of 3
skipped, context = get_skipped_context_pairs(corpus, 3)

# One-hot encode the skipped tokens and convert them into a NumPy array
encoded_skipped = [one_hot_encode(token, vocabulary) for token in skipped]
encoded_skipped = np.array(encoded_skipped)

# One-hot encode the context tokens and convert them into a NumPy array
encoded_context = [one_hot_encode(token, vocabulary) for token in context]
encoded_context = np.array(encoded_context)

# Convert the one-hot encoded arrays into PyTorch tensors with float32 data type
x = torch.tensor(encoded_skipped, dtype=torch.float32)
y = torch.tensor(encoded_context, dtype=torch.float32)


# Define a simple neural network module for the embedding layer.
class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear transformation from embedding dimension (10) back to vocabulary size
        self.linear = nn.Linear(len(vocabulary), 10)
    
    def forward(self, x):
        # Forward pass through the linear layer
        x = self.linear(x)
        return x


# Define a neural network module for the context layer.
class ContextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Linear transformation from embedding dimension (10) back to vocabulary size
        self.linear = nn.Linear(10, len(vocabulary))
    
    def forward(self, x):
        # Compute logits for each word in the vocabulary
        x = self.linear(x)
        return x
    
    
# Set the device to GPU if available, otherwise use the CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cpu = torch.device("cpu")

# Initialize the embedding and context layers and move them to the chosen device
embedding_layer = EmbeddingLayer().to(device)
context_layer = ContextLayer().to(device)

# Create Adam optimizers for both layers with a learning rate of 0.001
optim_embedding = optim.Adam(params=embedding_layer.parameters(), lr=0.001)
optim_context = optim.Adam(params=context_layer.parameters(), lr=0.001)

# Define the loss function: CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# Train the network with early stopping:
losses = []              # To record the loss per epoch.
best_loss = None         # To track the best (lowest) loss achieved.
patience = 20            # Maximum number of consecutive epochs without sufficient improvement.
patience_counter = 0     # Counter for epochs without sufficient improvement.
min_loss_improvement = 0.0001  # Minimum loss improvement to consider as progress.

# Training loop: run up to 15,000 epochs (early stopping may halt earlier).
for epoch in range(15000):
    # Move input and target tensors to the selected device (GPU/CPU)
    inputs, targets = x.to(device), y.to(device)
    
    # Convert one-hot encoded target vectors into class indices using argmax
    targets = targets.argmax(dim=1)
    
    # Forward pass: pass inputs through the embedding layer then the context layer
    outputs = context_layer(embedding_layer(inputs))
    
    # Compute the cross-entropy loss between predicted logits and target class indices
    loss = criterion(outputs, targets)
    
    # Zero the gradients for both optimizers
    optim_embedding.zero_grad()
    optim_context.zero_grad()
    
    # Backward pass: compute gradients with respect to the loss
    loss.backward()
    
    # Update the model parameters using the optimizers
    optim_embedding.step()
    optim_context.step()
    
    # Record the loss and print the current epoch's loss
    losses.append(loss.item())
        
    # Early stopping logic: check if the loss improved sufficiently.
    if best_loss is None or best_loss - loss.item() >= min_loss_improvement:
        best_loss = loss.item() # Update the best loss.
        best_model = (embedding_layer.state_dict(), context_layer.state_dict())
        # Reset the patience counter on improvement.
        patience_counter = 0
    else:
        # Increment counter if no sufficient improvement.
        patience_counter += 1
        
    # Print current epoch loss and the patience counter.
    print(f"epoch {epoch}: loss={loss.item()}; patience_counter: {patience_counter}")
    
    # If no improvement for 'patience' consecutive epochs, stop training.
    if patience_counter >= patience:
        print(f"The loss did not improve after {patience} consecutive epochs, stopping early.")
        break

# Load the best model parameters (i.e., those that yielded the lowest loss).
embedding_layer.load_state_dict(best_model[0])
context_layer.load_state_dict(best_model[1])

# Plot the training loss over epochs using seaborn.
sns.lineplot(x=list(range(len(losses))), y=losses)

# After training, extract the learned word embeddings for visualization
with torch.no_grad(): 
    # One-hot encode each word in the vocabulary
    encoded_vocabulary = [one_hot_encode(token, vocabulary) for token in vocabulary]
    encoded_vocabulary = torch.tensor(encoded_vocabulary, dtype=torch.float32)
    encoded_vocabulary = encoded_vocabulary.to(device)
    
    # Pass the one-hot vectors through the embedding layer to get dense embeddings
    embedding_vectors = embedding_layer(encoded_vocabulary).detach().to(cpu).numpy()

    # Use PCA to reduce the 10-dimensional embedding vectors to 2 dimensions for visualization
    pca = PCA(n_components=2, random_state=1)
    decomposed_vectors = pca.fit_transform(embedding_vectors)
    
    # Create an interactive scatter plot of the 2D embeddings using Plotly.
    # Each point represents a word, and hovering displays the corresponding vocabulary token.
    fig = px.scatter(x=decomposed_vectors[:,0], y=decomposed_vectors[:,1], hover_name=vocabulary)
    
    # Render the interactive plot
    plot(fig)
