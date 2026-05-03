**Livre Blanc Technique v1.0**

---

# Réseau P2P d’Inférence LLM avec Chargement Dynamique d’Adaptateurs LoRA  
### *Une plateforme décentralisée pour le raisonnement spécialisé et vérifiable*

**Statut :** Final pour MVP  
**Date :** Mai 2026  
**Auteurs :** Équipe du projet (synthèse des spécifications et des audits)

---

## Résumé

Ce livre blanc décrit l’architecture complète d’un réseau pair-à-pair (P2P) dédié à l’inférence de grands modèles de langage (LLM). Contrairement aux API centralisées comme ChatGPT, ce réseau permet de servir des milliers d’experts spécialisés (via des adaptateurs LoRA) en combinant un modèle de base fine-tuné pour le raisonnement complexe, un routage sémantique local, une distribution de contenu par BitSwap, et une couche de consensus économique sur blockchain de deuxième couche (L2). La rémunération des contributeurs (nœuds GPU, créateurs d’adaptateurs) et la confiance sont assurées par des micropaiements probabilistes, un système de preuve d’inférence par échantillonnage, et un marché de la qualité régulé par le staking. Le document détaille chaque couche technique, les mécanismes de sécurité, le modèle économique et la feuille de route pour un produit minimum viable (MVP) différenciant.

---

## 1. Introduction et Vision

Les grands modèles de langage centralisés (GPT-4, Claude, Gemini) excellent dans une multitude de tâches, mais ils souffrent de plusieurs limitations : coûts opaques, censure arbitraire, confidentialité inexistante, et incapacité à atteindre une expertise pointue dans des domaines spécifiques sans un paramétrage complexe. Par ailleurs, les créateurs de fine-tuning spécialisé n’ont aucun moyen direct de monétiser leur travail.

Notre vision est de créer un **réseau décentralisé d’inférence LLM** où :
- **La spécialisation** remplace la généralité : des adaptateurs LoRA, publiés par des experts, sont chargés dynamiquement pour répondre avec une précision inégalée à des problèmes de niche.
- **La qualité du raisonnement** est garantie par un modèle de base entraîné spécifiquement sur des données de raisonnement complexe, et par un marché incitatif qui récompense les meilleurs créateurs.
- **La confiance** est prouvée mathématiquement : provenance vérifiable, preuve d’inférence, chiffrement bout-en-bout.
- **Les utilisateurs et les fournisseurs de ressources** sont alignés économiquement via des micropaiements équitables et transparents, sans intermédiaire extractif.

Le réseau exploite du matériel grand public (RTX 3090/4090, 24 Go VRAM, SSD NVMe), une stack logicielle éprouvée (SGLang, libp2p, IPFS/BitSwap, Helios, Rust), et une couche économique sur rollup Ethereum (Arbitrum/Base). Ce livre blanc démontre la faisabilité technique et présente l’état finalisé de l’architecture à la suite de trois itérations correctives et d’un audit de sécurité.

---

## 2. Pourquoi un Réseau P2P Spécialisé ?

| Limitation des LLM généralistes | Solution apportée par le réseau |
|----------------------------------|----------------------------------|
| Réponses génériques, manque de profondeur métier | Routage sémantique vers le LoRA expert |
| Boîte noire : pas de preuve que le modèle a bien été exécuté | Proof of Inference par échantillonnage |
| Dépendance à un fournisseur central | Architecture entièrement décentralisée (nœuds, stockage, consensus) |
| Monétisation inexistante pour les créateurs de fine-tuning | 30 % des revenus de chaque requête reversés au créateur |
| Censure et politique d’utilisation opaque | Gouvernance par staking, pas d’autorité centrale |
| Confidentialité des données | Chiffrement E2E et option de nœuds certifiés |

---

## 3. Architecture Globale

Le système est divisé en **quatre couches asynchrones**, chacune indépendante et communiquant via des interfaces bien définies.

```
┌─────────────────────────────────────────┐
│          Couche Consensus (L2)          │
│  Smart Contracts, ZK-Coprocessors,      │
│  Réputation, Paiements, Slashing        │
└─────────────────────────────────────────┘
            ▲               │
            │               ▼
┌─────────────────────────────────────────┐
│       Couche Orchestration (Rust)       │
│  libp2p Swarm, Routage Sémantique HNSW, │
│  Helios Light Client, QoS, Tickets      │
└─────────────────────────────────────────┘
       ▲                       │
       │                       ▼
┌──────────────┐   ┌─────────────────────┐
│ Stockage     │   │  Couche Inférence   │
│ BitSwap /    │   │  SGLang + CUDA/      │
│ CID IPFS     │   │  Triton, KV Cache    │
│              │   │  Paging, LoRA        │
└──────────────┘   └─────────────────────┘
```

- **Couche Inférence (GPU)** : SGLang avec backend CUDA/Triton, gestion fine de la VRAM, chargement dynamique de LoRA.
- **Couche Orchestration (Middleware)** : Démon Rust intégrant libp2p, Helios, un index HNSW pour le routage sémantique, et les files d’attente QoS.
- **Couche Stockage** : Distribution des poids LoRA et du modèle de base via le protocole BitSwap, avec intégrité vérifiée par CID IPFS.
- **Couche Consensus** : Smart contracts sur L2 (Arbitrum/Base) pour le registre des adaptateurs, les paiements probabilistes, les preuves d’inférence, la réputation et le slashing.

---

## 4. Couche Inférence

### 4.1. Modèle de Base : Raisonneur Fondamental

Le nœud worker exécute un **modèle dense de 14 à 20 milliards de paramètres**, quantifié en INT4/AWQ, occupant ~9 Go de VRAM. Ce modèle n’est pas un LLM généraliste brut : il a été **fine-tuné spécifiquement pour le raisonnement complexe** sur le dataset [`open-thoughts/TaskTrove`](https://huggingface.co/datasets/open-thoughts/TaskTrove). L’entraînement ajoute des capacités de chaîne de pensée (Chain-of-Thought), de décomposition de tâches et d’auto-vérification.

- **Méthode** : Fine-tuning complet (ou fusion d’un LoRA de rang élevé) pour ne pas impacter la VRAM d’inférence.
- **Distribution** : Le modèle est packagé au format `safetensors`, identifié par un CID IPFS, et partagé via BitSwap comme n’importe quel adaptateur. Les nœuds le récupèrent avant de participer.

Ce méta-modèle garantit que même sans adaptateur spécialisé, le réseau fournit un raisonnement de meilleure qualité qu’un assistant générique.

### 4.2. Matériel Requis et Gestion de la VRAM

- **GPU** : NVIDIA RTX 3090/4090 (24 Go VRAM)
- **SSD** : NVMe PCIe 4.0, vitesse de lecture > 5 Go/s (obligatoire pour le swapping du KV Cache)
- **RAM système** : 64 Go

Allocation VRAM après chargement du modèle de base (9 Go) :
- **KV Cache** : jusqu’à 15 Go restants, gérés dynamiquement par un **Token Budgeting** avec `max_total_num_tokens = 18 000`.
- **LoRA multiplexés** : stockés majoritairement sur NVMe, chargés en VRAM selon un algorithme LRU (Least Recently Used). Si la VRAM est pleine, le LoRA le moins utilisé est déchargé.

### 4.3. KV Cache Paging

Lorsque le budget de tokens est saturé et qu’une requête prioritaire (Fast Lane) arrive, les requêtes Spot Lane en cours voient leur cache KV **déchargé sur le SSD NVMe** (swap-out). Elles sont suspendues et reprises plus tard (swap-in). Ce mécanisme est activé par le *KV Cache Paging* de SGLang. La latence de swap est le risque principal ; elle doit être validée expérimentalement (< 20 ms par opération pour rester transparent).

### 4.4. Multiplexage LoRA

- Chargement dynamique via l’API SGLang (`--lora-paths`).
- Rang (r) maximum fixé à 64, taille de fichier maximale 150 Mo.
- Temps de commutation en VRAM < 50 ms.
- Les poids LoRA sont pré-validés à l’étape de téléchargement (voir Sécurité).

---

## 5. Couche Orchestration (Middleware Rust)

Le démon Rust est le cerveau du nœud. Il exécute plusieurs modules en parallèle, communiquant via des canaux asynchrones pour une robustesse maximale.

### 5.1. Réseau P2P : libp2p Private Swarm

- **Topologie** : libp2p en mode « private swarm » ; l’accès est réservé aux nœuds ayant staké et prouvant la possession d’un GPU éligible.
- **Découverte de pairs** : Kademlia DHT, mais uniquement pour localiser les fournisseurs d’un CID (les pairs sont déjà autorisés).
- **Protocoles de transport** : TCP/QUIC optimisés pour les transferts de blocs de 150 Mo.

### 5.2. Client Léger L2 : Helios

Helios est compilé dans le binaire Rust. Il vérifie de manière trustless les racines d'état du rollup L2. Ainsi, le nœud lit les événements du Smart Contract Registre (publication de LoRA, mise à jour de réputation) sans dépendre d’un RPC centralisé.

### 5.3. Routage Sémantique avec HNSW

**Objectif** : Diriger chaque prompt utilisateur vers le LoRA le plus pertinent.

1. Le prompt est vectorisé par un modèle d’embedding léger (`bge-small-en-v1.5`, < 100 Mo, exécution CPU < 2 ms).
2. Une recherche k-NN dans un index **HNSW local** retourne le CID du LoRA.
3. L’index est mis à jour en continu à partir des événements du Smart Contract (nouveaux LoRA, ajustements de réputation).

**Ajustement dynamique** : Si un LoRA reçoit des votes négatifs répétés (RLHF), son vecteur est pénalisé localement, réduisant sa visibilité.

### 5.4. Files d’Attente et QoS

- **Fast Lane (B2B)** : Requêtes signées prouvant un staking B2B. Priorité absolue.
- **Spot Lane (B2C)** : Requêtes gratuites, avec timeout de 5 secondes. Elles peuvent être suspendues si la VRAM est pleine.

### 5.5. Cache Miss et Récupération de LoRA

Si le LoRA demandé n’est pas dans le cache NVMe, le nœud initie un téléchargement BitSwap auprès des pairs. La cible de latence est **< 300 ms** pour un fichier de 150 Mo. En parallèle, un pré-fetch automatique des 50 LoRAs les plus populaires est effectué pendant les périodes d’inactivité GPU.

---

## 6. Couche Stockage (BitSwap et IPFS)

- **Format** : Tous les artefacts (modèle de base, LoRA) sont au format Safetensors, garantissant une structure validable et sans exécution arbitraire.
- **Identification** : Chaque fichier possède un CID (Content Identifier) IPFS basé sur SHA-256.
- **Distribution** : Le protocole BitSwap permet un téléchargement multi-sources (Swarm). La vérification d’intégrité est native : un bloc reçu ne correspondant pas au CID est rejeté automatiquement.

---

## 7. Couche Consensus et Modèle Économique

### 7.1. Smart Contracts sur L2

Deux contrats principaux sont déployés sur Arbitrum/Base :
1. **Registre LoRA** : stocke `[CID, creator, embeddingVector, baseModelCID, reputationScore, stakedAmount]`. Les créateurs paient un stake de ~5 $ pour publier.
2. **Canal de Paiement / Tickets** : gère la soumission des tickets probabilistes, la redistribution automatique (70 % nœud, 30 % créateur) et le slashing.

### 7.2. Micropaiements Probabilistes

Afin d’éviter une transaction par token, un schéma de tickets probabilistes est utilisé :

- L’inférence est découpée en chunks de 10 tokens.
- Avant que le nœud ne génère un chunk, le client envoie un ticket cryptographique (signé) couvrant ce chunk.
- Le nœud vérifie la signature et un prédicat : `hash(ticket) % 1000 == 0` détermine si le ticket est gagnant.
- Les tickets gagnants valent 1000× le montant unitaire et sont soumis au smart contract pour paiement.
- Les tickets perdants sont simplement jetés.

Ce mécanisme réduit les coûts on-chain de 99,9 %. La redistribution de 30 % au créateur du LoRA (identifié par le CID présent dans le ticket) est automatique.

### 7.3. Modèle Économique Flywheel

- **Pilier 1 – Proof of Availability (PoA)** : Les nœuds GPU reçoivent une émission de tokens pour leur uptime, mesurée par des pings signés périodiques.
- **Pilier 2 – Staking B2B** : Les entreprises stakent des tokens pour obtenir une bande passante garantie (Fast Lane). La demande de staking augmente la valeur du token.
- **Pilier 3 – Gatekeeper RLHF (B2C)** : Les utilisateurs gratuits effectuent des tâches RLHF (votes comparatifs) pour gagner des JWT d’accès. Les données RLHF sont valorisées et revendues.

### 7.4. Réputation et Ajustement des LoRA

- Le score de réputation d’un créateur évolue selon les votes RLHF et les résultats des honeypots.
- Un score bas peut entraîner une sortie de l’index HNSW et la perte du stake.
- La transparence est totale : chaque réponse affiche le CID, le créateur et son score.

---

## 8. Sécurité et Intégrité

### 8.1. Validation des Poids (Anti-Poisoning)

Avant tout chargement en VRAM, le fichier Safetensors est scanné :
- Vérification SHA-256 par rapport au CID.
- Détection de NaN, Inf et de normes L2 aberrantes (seuil dynamique = moyenne des couches du modèle de base + 20 %).

### 8.2. Isolation des Prompts

Le démon Rust force l’encapsulation du prompt dans le template strict du modèle de base (ex: ChatML). Le LoRA ne peut pas modifier le format ni accéder à d’autres parties du système.

### 8.3. Chiffrement Bout-en-Bout

Les clients B2B peuvent chiffrer leurs prompts avec la clé publique du nœud cible. Le nœud déchiffre en mémoire GPU avant l’inférence. Pour les données très sensibles, des **Nœuds Certifiés** (stake massif, audits de sécurité) sont proposés.

### 8.4. Preuve d’Inférence (PoI) par Échantillonnage

- **Validateurs** : Une catégorie de nœuds qui ne font pas d’inférence mais valident les réponses.
- Une requête sur 1000 est envoyée à 3 workers différents. Le validateur compare les réponses avec un modèle Cross-Encoder léger (MiniLM-L6-v2, 80 Mo).
- Si un worker diverge au-delà d’un seuil de similarité, le validateur soumet une preuve cryptographique au contrat L2. Le worker est slasé (perte de stake), et le validateur reçoit une récompense.

### 8.5. Résistance aux Bots (Gatekeeper RLHF)

- Les utilisateurs gratuits doivent résoudre une courte preuve de travail dans le navigateur avant de voter.
- 10 % des tâches RLHF sont des **honeypots** dont la réponse est connue. Des échecs répétés entraînent la révocation du JWT.

---

## 9. Spécialisation et Différenciation (Patch V4)

### 9.1. Catalogue de Lancement MVP

Pour amorcer le réseau, un ensemble de **20 à 30 LoRA d’élite** est sélectionné et validé par l’équipe fondatrice. Ces adaptateurs couvrent des niches exigeantes : programmation, mathématiques, droit, médecine, conseil financier, etc. Les créateurs sont invités à publier avec des incitations initiales.

### 9.2. Transparence de Provenance

Chaque réponse du réseau inclut :
- CID du LoRA utilisé
- Adresse du créateur (ENS)
- Score de réputation on-chain
- Montant staké

Cette signature de provenance permet à l’utilisateur de vérifier l’origine de l’expertise et de faire confiance – ou non – à la source.

### 9.3. Mode Benchmark Direct

Un endpoint de comparaison envoie la même requête au réseau P2P **et** à une API centralisée (ChatGPT, Claude). Les deux réponses sont présentées anonymement ; l’utilisateur vote pour la meilleure. Ces votes enrichissent le dataset RLHF, fournissent un retour permanent sur la qualité, et constituent un outil marketing puissant pour démontrer la supériorité du réseau.

Le mode benchmark est activable sur les requêtes Spot Lane.

---

## 10. Feuille de Route et MVP

**Phase 0 – Validation Expérimentale (3 mois)**
- Mesure de la latence de swap KV Cache sur SSD NVMe (critique).
- Prototype centralisé du moteur SGLang avec budget tokens et LRU.
- Test de bout en bout avec un petit swarm libp2p.

**Phase 1 – MVP Core (6 mois)**
- Déploiement du middleware Rust avec Helios, HNSW, et files QoS.
- Intégration des tickets probabilistes minimalistes (paiements hors chaîne simulés).
- Lancement du catalogue initial de 25 LoRA.
- Interface web cliente avec affichage de provenance.

**Phase 2 – Réseau Décentralisé et L2 (6-9 mois)**
- Déploiement des smart contracts sur testnet L2.
- Activation du staking et du slashing.
- Mise en place des nœuds validateurs et du PoI.
- Mode benchmark connecté aux API externes.

**Phase 3 – Lancement Public**
- Migration sur mainnet.
- Ouverture du registre à tous les créateurs.
- Gouvernance progressive par DAO.

---

## 11. Risques et Mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Latence de swap KV Cache trop élevée | Moyenne | Très haut | Phase 0 dédiée ; si la latence > 50 ms, utiliser un modèle plus petit ou exiger 32 Go VRAM. |
| Complexité du middleware Rust | Haute | Élevé | Architecture modulaire interne, tests intensifs, embauche de contributeurs expérimentés. |
| Faible adoption initiale (nœuds / clients) | Moyenne | Élevé | Catalogue initial attractif, incitations financières (PoA), marketing ciblé vers les communautés open-source. |
| Coûts des preuves ZK (PoI) | Faible | Moyen | Le PoI par échantillonnage évite la ZK lourde ; seuls les litiges passent par une preuve légère. |
| Volatilité du prix du token | Élevé | Moyen | Mécanismes de stabilisation (staking dynamique), focus sur l’utilité réelle ; la valeur repose sur les services rendus. |
| Empoisonnement sémantique du registre | Faible | Élevé | Stake-to-publish, ajustement local par votes RLHF, système de réputation. |

---

## 12. Conclusion

Le réseau P2P d’inférence LLM présenté ici offre une alternative crédible aux silos centralisés de l’IA. En combinant une architecture modulaire éprouvée, une monétisation équitable et des mécanismes de confiance décentralisés, il rend possible l’émergence d’un écosystème mondial où les experts sont rémunérés pour leurs contributions, les utilisateurs obtiennent des réponses d’une précision chirurgicale, et les fournisseurs de calcul sont justement compensés.

Le chemin vers le MVP est balisé, les points de risque sont identifiés et maîtrisés, et chaque brique technologique a fait ses preuves indépendamment. La prochaine étape est la validation expérimentale de la latence mémoire, qui conditionnera le rythme du déploiement. Une fois cette barrière franchie, le réseau peut être assemblé rapidement et offrir au monde une intelligence décentralisée, spécialisée et vérifiable.

---

*Ce livre blanc reflète la version finale de l’architecture après les patches V2, V3 et V4, et l’audit technique. Toute correspondance : [adresse de contact].*