# MiniQuartoGPT

A Shakespeare-specialized GPT built on nanoGPT with enhanced features for Early Modern English text generation
QuartoGPT is a miniaturized transformer model specifically designed for generating Elizabethan text with improved metrical and stylistic accuracy. Built upon Andrej Karpathy's nanoGPT framework, it incorporates novel architectural enhancements for poetry and verse generation.

This ongoing work is testing the model's capabilities on various hardware configurations, using both basic and advanced setups. A core part of this process involves the implementation of key features such as Meter-Aware Positional Embeddings and a suite of Customized Loss Functions. Specifically, we are utilizing the MetricalLoss function to penalize deviations from proper syllable counts and integrating the RhymeAwareLoss function to reward consistent rhyme schemes. This approach allows us to rigorously validate the model's performance in generating text that is both  authentic in the style of generation and metrically precise.
