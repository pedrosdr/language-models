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

# Define a corpus of sentences covering Computer Science and Civil Engineering
corpus = [
    "Computer science drives innovation through efficient algorithms and programming languages.",
    "Civil engineering designs safe and sustainable structures for modern cities.",
    "Software development in computer science supports automated systems in civil engineering projects.",
    "Computer algorithms optimize construction schedules and resource allocation in civil engineering.",
    "Data analysis in computer science helps predict infrastructure performance and safety.",
    "Civil engineers use advanced software to model stress and load on bridges.",
    "Computer science innovations lead to better simulation tools for civil engineering design.",
    "Robust algorithms ensure reliability in both software systems and civil infrastructure monitoring.",
    "Computer science supports the development of smart sensors for monitoring structural health.",
    "Civil engineering integrates modern materials with technology to build resilient structures.",
    "Advanced computing models simulate earthquake impacts on civil infrastructure.",
    "Computer science techniques enhance the design and analysis of civil structures.",
    "Civil engineering projects benefit from data analytics and machine learning algorithms.",
    "Simulation software developed in computer science aids in stress testing bridges.",
    "Computer programming skills are essential for modern civil engineering research.",
    "Civil engineering uses geographic information systems powered by computer algorithms.",
    "Big data in computer science improves decision-making in large-scale construction projects.",
    "Civil engineers rely on software simulations to predict traffic flow and bridge behavior.",
    "Machine learning in computer science analyzes structural data from sensors in buildings.",
    "Civil engineering designs are enhanced by computer-aided drafting and modeling software.",
    "Computer networks enable real-time monitoring of civil infrastructure and urban systems.",
    "Civil engineering relies on statistical models developed through computer science to ensure safety.",
    "Algorithm development in computer science speeds up simulation processes in civil engineering tasks.",
    "Civil engineering and computer science collaborate to create innovative, sustainable urban solutions.",
    "Data visualization tools from computer science help civil engineers understand project progress.",
    "Computer simulations predict soil behavior and stress distribution in civil construction projects.",
    "Civil engineering standards are updated using insights from computer-based data analysis.",
    "Computer science improves infrastructure planning with high-speed data processing and smart algorithms.",
    "Civil engineers integrate sensors and computer systems to monitor bridge health continuously.",
    "Artificial intelligence in computer science enhances the efficiency of civil engineering designs.",
    "Civil engineering projects benefit from computer-aided structural analysis and real-time monitoring systems.",
    "Computer science methodologies simplify complex calculations for civil engineering load assessments.",
    "Civil engineers use digital models created by computer science to simulate water flow in dams.",
    "Computer programming enables the automation of monitoring systems in large civil engineering projects.",
    "Civil engineering relies on simulation software from computer science for earthquake and flood analysis.",
    "Computer science research drives innovations that improve the materials used in civil engineering construction.",
    "Civil engineering designs incorporate computational tools from computer science for precise planning.",
    "Machine learning models in computer science predict the longevity of civil infrastructure components.",
    "Civil engineering safety measures improve with computer-driven data analysis and predictive modeling.",
    "Computer science contributes to civil engineering by optimizing construction management and workflow scheduling.",
    "Civil engineers design energy-efficient buildings with assistance from computer simulation and thermal analysis.",
    "Computer science tools automate routine tasks, freeing civil engineers to focus on innovative design challenges.",
    "Civil engineering uses digital mapping and computer-aided design to create detailed construction blueprints.",
    "Computer vision algorithms help analyze structural images and detect issues in civil engineering projects.",
    "Civil engineers depend on reliable software developed through computer science for design and project management.",
    "Computer science innovations enhance virtual reality training for civil engineering workers on construction sites.",
    "Civil engineering challenges are solved using computer algorithms that optimize design parameters and improve efficiency.",
    "Computer science principles support the development of smart cities with efficient civil engineering infrastructures.",
    "Civil engineering integrates environmental analysis and computer models to design sustainable urban areas.",
    "Computer science and civil engineering together innovate, creating advanced solutions for construction, sustainability, and urban development."
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
