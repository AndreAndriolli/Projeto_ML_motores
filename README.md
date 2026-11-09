# Diagnóstico de danos em rolamentos com corrente do motor e vibração

Pipeline completo de machine learning para classificar o estado de rolamentos (saudável, dano no anel externo, dano no anel interno) a partir de sinais de **corrente do motor (MCS)** e **vibração**, usando o dataset da Universidade de Paderborn (KAt-DataCenter). O projeto reproduz e estende o estudo de Lessmeier et al. (PHME 2016).

**Destaques**
- Extração paralela de features no tempo, na frequência, por Wavelet Packet e por espectro de envelope (BPFO/BPFI).
- Validação **por rolamento** (nenhum rolamento aparece no treino e no teste), com seleção de features ajustada só no treino de cada fold.
- Sete algoritmos do artigo (CART, RF, BT, NN, SVM, ELM, kNN) e um ensemble por votação.

**Principais resultados**
- Classificação em 3 classes com vibração: 73 a 77 % (o artigo relata 98 %).
- Detecção binária (saudável × danificado) com todos os rolamentos e condições: 87,7 % de acurácia balanceada.
- A vibração supera a corrente, como no artigo. A corrente não generalizou para rolamentos não vistos na classificação em 3 classes.

**Estrutura**
- `rolamentos.py`: módulo com leitura, features, seleção, modelos e avaliação
- `01_extracao.ipynb`: extração de features dos arquivos `.mat`
- `02_dataset_analise.ipynb`: rótulos e análise exploratória
- `03_modelagem.ipynb`: experimentos e comparação com o artigo

site do estudo e dados:
https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download?utm_source=chatgpt.com
