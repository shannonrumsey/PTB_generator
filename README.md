# PTB Generator

```plaintext
├── requirements.txt
└── run.py
```
A decoder-only language model is created to generate sentences from the Penn Treebank dataset. This model achieves a perplexity of 269, compared to GPT-2 which gets around 65. Data is imported in the run.py from Hugging Face. 

To run:

```bash
pip install -r requirements.txt
python run.py
```
