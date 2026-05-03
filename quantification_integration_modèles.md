Voici les instructions détaillées pour quantifier vous-même GPT-OSS 20B ou Qwen 3 14B en AWQ INT4, prêtes à être intégrées à la documentation technique du projet.

---

## Annexe Technique – Quantification AWQ INT4 des Modèles de Base

Cette section décrit la procédure pour quantifier en AWQ 4-bit les modèles GPT-OSS 20B et Qwen 3 14B, en utilisant **AutoRound** (recommandé pour sa qualité) ou l’outil intégré de SGLang. Les poids résultants sont au format **safetensors**, directement utilisables avec SGLang et compatibles avec la couche de sécurité (validation SHA-256, scan NaN/Inf, norme L2).

### Pourquoi AWQ ?

- **Performance GPU** : kernels CUDA optimisés (Marlin) → vitesse d’inférence proche du FP16.
- **Qualité** : préservation des activations importantes, dégradation minime (~0.5-1% de précision en moins) par rapport au modèle original.
- **Format** : produit des fichiers `.safetensors` standard, chargeables nativement par SGLang sans conversion.
- **Compatibilité LoRA** : le modèle quantifié AWQ accepte le chargement dynamique d’adaptateurs LoRA (contrairement à certaines variantes GPTQ).

### Prérequis

- GPU NVIDIA avec ≥ 24 Go VRAM (RTX 3090/4090 recommandé)
- Python ≥ 3.10, PyTorch ≥ 2.1, CUDA ≥ 12.1
- Espace disque : ~40 Go pour le modèle original + quantifié

### Option 1 – Utilisation d’AutoRound (recommandé)

[AutoRound](https://github.com/intel/auto-round) produit une quantification AWQ de haute qualité en optimisant les arrondis des poids.

#### Installation

```bash
pip install auto-round
pip install transformers accelerate safetensors
```

#### Quantification de GPT-OSS 20B

```python
from auto_round import AutoRound
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai/gpt-oss-20b"  # ou "Qwen/Qwen3-14B"
output_dir = "./gpt-oss-20b-awq-int4"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Configuration AWQ INT4
round = AutoRound(
    model,
    tokenizer,
    bits=4,
    group_size=128,
    scheme="asym",          # asym = asymétrique (meilleure qualité)
    weight_config={"*": {"data_type": "int"}}
)

# Lancer la quantification (prend quelques heures sur GPU 24 Go)
round.quantize()
round.save_quantized(output_dir, format="auto", safetensors=True)

tokenizer.save_pretrained(output_dir)
print(f"Modèle quantifié sauvegardé dans : {output_dir}")
```

#### Quantification de Qwen 3 14B ou Qwen 2.5 14B

```python
# Mêmes étapes en remplaçant model_name par "Qwen/Qwen3-14B"
model_name = "Qwen/Qwen3-14B"
output_dir = "./qwen3-14b-awq-int4"

# ...
```

#### Vérification rapide

```bash
python -m sglang.launch_server --model ./gpt-oss-20b-awq-int4 --quantization awq
```

Puis envoyez une requête de test pour contrôler la qualité.

### Option 2 – Modèles pré-quantifiés (gain de temps)

Des communautés ont déjà quantifié ces modèles en AWQ INT4. Vous pouvez les télécharger directement depuis HuggingFace, puis les packager pour le réseau.

| Modèle de base | Modèle AWQ disponible | Lien |
| :--- | :--- | :--- |
| GPT-OSS 20B | `marcsun13/gpt-oss-20b-awq` (exemple, vérifier l’actualité) | HuggingFace |
| Qwen 2.5 14B | `Qwen/Qwen2.5-14B-Instruct-AWQ` (officiel) | [lien](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ) |
| Qwen 3 14B | `Qwen/Qwen3-14B-AWQ` (officiel prévu) | Surveiller le repo Qwen |

Si vous utilisez un modèle pré-quantifié, assurez-vous qu’il soit bien au format AWQ (config.json contient `"quantization_config": {"quant_method": "awq"}`) et que les poids soient en safetensors. Il vous suffit de le télécharger et de le distribuer via IPFS.

### Option 3 – Quantification avec l’outil SGLang (si disponible)

SGLang peut appliquer une quantification AWQ à la volée lors du chargement, mais cela ne crée pas de checkpoint sauvegardé. Pour créer un artefact réutilisable, utilisez plutôt AutoRound ou téléchargez un modèle pré-quantifié. Toutefois, pour un test rapide :

```bash
python -m sglang.launch_server \
    --model openai/gpt-oss-20b \
    --quantization awq \
    --dtype auto
```

Cela quantifie en mémoire, mais le modèle disparaît à l’arrêt du serveur.

### Intégration dans le pipeline du réseau P2P

Une fois le modèle quantifié obtenu (dossier avec safetensors, config.json, tokenizer), il doit être :

1. **Packagé** : tous les fichiers sont ajoutés à un répertoire et hachés avec IPFS (CID racine calculé sur l’ensemble).
2. **Distribué** : le CID est enregistré dans le Smart Contract du registre comme `baseModelCID` obligatoire pour les nœuds workers.
3. **Validé** : chaque nœud vérifie le SHA-256 du dossier complet (ou des fichiers individuels) par rapport au CID avant de charger SGLang.

Exemple de calcul du CID sur le répertoire :

```bash
ipfs add -r ./gpt-oss-20b-awq-int4 --cid-version=1
# retourne le CID racine
```

### Dépannage

- **Manque de mémoire** : si la quantification échoue par OOM, utilisez `device_map="cpu"` pour placer le modèle original sur CPU et quantifier avec CPU (très lent mais possible).
- **Group_size** : 128 est un bon compromis qualité/taille. La VRAM occupée sera d’environ 9-11 Go pour un modèle 14-20B.
- **Schema asym** : recommandé pour préserver la qualité des activations.

---

Ces instructions sont prêtes à être insérées dans la documentation du projet, en complément de la section « Couche Inférence ».