# CP4285-G1

## Content-Based Model Progress
- Tried out a simple two tower model + kNN model, compared to a baseline centroid model that simply averages the multimodal feature vectors of songs played by a user (this becomes the "user embedding"). Results are pretty middling, not much improvements over the CF model for warm items, but we can see decent (?) performance for cold items. Relatively speaking performance is fine, but overall metrics are kinda bad (could be a symptom of a hard recommendation problem)
- Did not experiment much with the underlying two tower architecture. Currently just using a simple two layer linear model for both user and item embeddings. Loss curve seems to indicate signs of underfitting (loss does not decrease by much over the course of training), so we probably want to increase model complexity and test again
- Best to use GPU to train the two tower models. With grid search over different hyperparameters, training becomes pretty long on CPU.